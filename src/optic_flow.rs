use std::{f64::consts::PI, fmt::Debug, ops::Range};

use kv::{Bucket, Integer, Json};
use serde::{Deserialize, Serialize};
use nalgebra::{Point2, Rotation};
use itertools::{Itertools, izip};
use log::trace;
use measure_time::*;
use opencv::{core::{CV_32FC2, Point2f, Point2i, Scalar, Size2i, TermCriteria, TermCriteria_Type, no_array}, highgui::imshow, imgproc::{COLOR_BGR2GRAY, FONT_HERSHEY_SIMPLEX, INTER_AREA, LINE_8, LINE_AA, good_features_to_track}, prelude::{Mat, MatExprTrait, MatTrait, MatTraitManual}, video::calc_optical_flow_pyr_lk};
use statrs::statistics::Statistics;
use thiserror::Error;

use crate::{camera::{CameraError, Lens}, util::sliding_average, video_source::{VideoSource, VideoSourceError}};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct OpticFlowPoint {
    #[serde(with = "point_serde")]
    pub prev: Point2<f32>,
    #[serde(with = "point_serde")]
    pub curr: Point2<f32>,
    pub error: f32,
}

mod point_serde {
    use nalgebra::Point2;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(
        point: &Point2<f32>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> where S: Serializer {
        (point.coords.x, point.coords.y).serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<Point2<f32>, D::Error> where D: Deserializer<'de> {
        let coords: (f32, f32) = Deserialize::deserialize(deserializer)?;
        Ok(Point2::new(coords.0, coords.1))
    }
}

/// An optic flow between the previous frame and the current one
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FrameOpticFlow {
    pub points: Vec<OpticFlowPoint>,
    pub rms_error: f64,
}

impl FrameOpticFlow {
    pub fn new(points: Vec<OpticFlowPoint>) -> Self {
        let errors = points.iter().map(|p| p.error as f64);
        let rms_error = Statistics::quadratic_mean(errors.clone());

        Self {
            points, rms_error
        }
    }

    // pub fn good_points(&self) -> Vec<usize> {
    //     let mut good_points = Vec::new();

    //     let mut distances = self.points.iter().map(|p| p.distance() ).collect_vec();
    //     distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    //     let distance_median = distances[distances.len() / 2];
        
    //     let distance_threshold = distance_median.min(10.0);

    //     for (ix, distance) in distances.iter().copied().enumerate() {
    //         if distance >= distance_threshold { 
    //             good_points.push(ix);
    //         }
    //     }

    //     good_points
    // }

    pub fn prev_points(&self) -> Mat {
        let mut prev_points = Mat::zeros(self.points.len() as i32, 1, CV_32FC2).unwrap().to_mat().unwrap();

        for (ix, point) in self.points.iter().enumerate() {
            *prev_points.at_mut(ix as i32).unwrap() = Point2f::new(point.prev.coords.x, point.prev.coords.y);
        }

        prev_points
    }

    pub fn curr_points(&self) -> Mat {
        let mut curr_points = Mat::zeros(self.points.len() as i32, 1, CV_32FC2).unwrap().to_mat().unwrap();

        for (ix, point) in self.points.iter().enumerate() {
            *curr_points.at_mut(ix as i32).unwrap() = Point2f::new(point.curr.coords.x, point.curr.coords.y);
        }

        curr_points
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FramePose {
    pub rotation: Rotation<f64, 3>,
    pub points_used: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct FrameRotation {
    /// radians/s
    pub roll: f64,
    /// radians/s
    pub pitch: f64,
    /// radians/s
    pub yaw: f64,
    pub time: f64,
    pub quality: f64,
}

#[derive(Debug, Error)]
pub enum OpticFlowRotationError {
    #[error("couldn't convert to greyscale, {0}")]
    ConvertToGreyscale(opencv::Error),
    #[error("video source error, {0}")]
    VideoSource(#[from] VideoSourceError),
    #[error("couldn't find features to track, {0}")]
    GoodFeaturesToTrack(opencv::Error),
    #[error("couldn't calculate optic flow, {0}")]
    CalculateOpticFlow(opencv::Error),
    #[error("couldn't convert tracked points data, {0}")]
    PointsDataFormat(opencv::Error),
    #[error("OpenCV error, {0}")]
    OpenCV(#[from] opencv::Error),
    #[error("camera error, {0}")]
    Camera(CameraError),
}

pub struct Retracker {
    min_quality_at_pts_used: usize,
    max_quality_at_pts_used: usize,
    min_qty: usize,
    min_qty_rel: f32,
    min_used_qty: usize,
    min_used_qty_rel: f32,
    max_base_qty: usize,
    max_err_rel: f64,
    max_t: f64,
    should_retrack: bool,
    retracking: bool,
    current_frame: usize,
    current_time: f64,
    current_quality: f64,
    base_qty: Option<usize>,
    base_err: Option<f64>,
    base_t: Option<f64>,
    base_ix: usize,
}

impl Default for Retracker {
    fn default() -> Self {
        Self::new(80, 0.7, 60, 0.6, 200, 2.0, 2.0)
    }
}

impl Retracker {
    pub fn new(
        min_qty: usize,
        min_qty_rel: f32,
        min_used_qty: usize,
        min_used_qty_rel: f32,
        max_base_qty: usize,
        max_err_rel: f64,
        max_t: f64,
    ) -> Self {
        Self {
            min_quality_at_pts_used: min_used_qty,
            max_quality_at_pts_used: max_base_qty,
            min_qty,
            min_qty_rel,
            min_used_qty,
            min_used_qty_rel,
            max_base_qty,
            max_err_rel,
            max_t,
            should_retrack: true,
            retracking: false,
            current_frame: 0,
            current_time: 0.0,
            current_quality: 1.0,
            base_qty: None,
            base_err: None,
            base_t: None,
            base_ix: 0,
        }
    }

    pub fn got_frame(&mut self, ix: usize, time: f64) {
        self.current_frame = ix;
        self.current_time = time;
        self.retracking = false;
        self.current_quality = 1.0;

        if let Some(base_t) = self.base_t {
            if time - base_t > self.max_t {
                trace!("Will retrack because too much time has passed");
                self.should_retrack = true;
            }
        }
    }

    pub fn should_retrack(&self) -> bool {
        self.should_retrack && !self.retracking
    }

    pub fn force_retrack_next(&mut self) {
        self.should_retrack = true;
    }

    fn is_optic_flow_good(&self, flow: &FrameOpticFlow) -> bool {
        let points = flow.points.len();
        let errors = flow.points.iter().map(|p| p.error as f64);
        let rms_error = Statistics::quadratic_mean(errors.clone());

        if points < self.min_qty {
            trace!("Will retrack because too few points left");
            return false;
        }

        if let Some(base_qty) = self.base_qty {
            if (points as f32) < (self.min_qty_rel * base_qty as f32) {
                trace!("Will retrack because too few points left relative to the base track");
                return false;
            }
        } else {
            return false;
        }
        if let Some(base_err) = self.base_err {
            if rms_error > self.max_err_rel * base_err {
                trace!("Will retrack because error grew too large relative to the base error");
                return false;
            }
        } else {
            return false;
        }

        true
    }

    pub fn got_optic_flow(&mut self, flow: &FrameOpticFlow) {
        self.current_quality *= (1.0 / (flow.rms_error / 2.0)).clamp(0.5, 1.0);

        if self.retracking {
            self.base_err = Some(flow.rms_error);
            self.base_qty = Some(flow.points.len().min(self.max_base_qty));
        } else if !self.is_optic_flow_good(flow) {
            self.should_retrack = true;
        }
    }

    fn is_pose_good(&self, pose: &FramePose) -> bool {
        let points = pose.points_used;

        if points < self.min_used_qty {
            trace!("Will retrack because too few points used recovering camera pose");
            return false;
        }
        if let Some(base_qty) = self.base_qty {
            if (points as f32) < (self.min_used_qty_rel * base_qty as f32) {
                trace!("Will retrack because too few points used recovering camera pose relative to the base track");
                return false;
            }
        } else {
            return false;
        }

        true
    }

    pub fn got_pose(&mut self, pose: &FramePose) {
        let quality = (pose.points_used - self.min_quality_at_pts_used) as f64 / (self.max_quality_at_pts_used - self.min_quality_at_pts_used) as f64;
        let quality = quality.clamp(0.0, 1.0);
        self.current_quality *= quality;

        if self.retracking {
        } else if !self.is_pose_good(pose) {
            self.should_retrack = true;
        }
    }

    pub fn retracking(&mut self) {
        self.should_retrack = false;
        self.retracking = true;
        self.base_ix = self.current_frame;
        self.base_t = Some(self.current_time);
    }
}

pub struct DebugOptions {
    pub scale: f32,
}

pub struct OpticFlowRotation<'cache> {
    src: VideoSource,
    lens: Lens,
    frames: Vec<Option<FrameRotation>>,
    debug: Option<DebugOptions>,
    cache: Bucket<'cache, Integer, Json<FrameRotation>>,
}

impl <'cache> OpticFlowRotation<'cache> {
    pub fn new(src: VideoSource, lens: Lens, debug: Option<DebugOptions>, cache: Bucket<'cache, Integer, Json<FrameRotation>>) -> Self {
        let mut frames = vec![None; src.frame_count().max(1) - 1];
        for item in cache.iter() {
            let item = item.unwrap();
            let k: usize = item.key().unwrap();
            let v = item.value::<Json<FrameRotation>>().unwrap();
            let v = v.0;
            frames[k] = Some(v);
        }

        Self { src, lens, frames, debug, cache }
    }

    pub fn recalc_for_ix_range_if_needed(&mut self, ix_range: Range<usize>) -> Result<(), OpticFlowRotationError> {
        let mut ix_range = ix_range;
        ix_range.end = ix_range.end.min(self.frames.len());

        let mut ranges_to_recalc = vec![];
        for (missing, frames) in &self.frames[ix_range.clone()].iter().copied().zip(ix_range).group_by(|(r, _ix)| r.is_none()) {
            if missing {
                let mut frames = frames.into_iter();
                let first_frame_ix = frames.next().unwrap().1;
                let last_frame_ix = frames.last().map(|(_, ix)| ix).unwrap_or(first_frame_ix);

                ranges_to_recalc.push(first_frame_ix..last_frame_ix+1);
            }
        }

        for range_to_recalc in ranges_to_recalc {
            self.recalc_for_ix_range(range_to_recalc)?;
        }
        
        Ok(())
    }

    pub fn recalc_for_ix_range(&mut self, ix_range: Range<usize>) -> Result<(), OpticFlowRotationError> {
        let mut retracker = Retracker::default();
        let mut ix_range = ix_range;
        ix_range.end = ix_range.end.min(self.frames.len());

        let mut src_frame = Mat::default();
        let mut frame_1_grey = Mat::default();
        let mut frame_2_grey = Mat::default();

        let prev_frame = &mut frame_1_grey;
        let curr_frame = &mut frame_2_grey;
        let mut prev_time = self.src.get_frame(ix_range.start, &mut src_frame)?;
        opencv::imgproc::cvt_color(&src_frame, curr_frame, COLOR_BGR2GRAY, 0)
            .map_err(OpticFlowRotationError::ConvertToGreyscale)?;

        let mut curr_points = Mat::default();

        let width = self.lens.dimensions.w;
        let height = self.lens.dimensions.h;
        let min_x = width as f32 * 0.05;
        let max_x = width as f32 * 0.95;
        let min_y = height as f32 * 0.05;
        let max_y = height as f32 * 0.95;

        'next_frame: for ix in ix_range {
            let time = {
                debug_time!("get_frame");
                if let Ok(time) = self.src.get_frame(ix + 1, &mut src_frame) {
                    time
                } else {
                    retracker.force_retrack_next();
                    continue 'next_frame;
                }
            };
            retracker.got_frame(ix, time);

            std::mem::swap(prev_frame, curr_frame);
            let frame_duration = time - prev_time;
            prev_time = time;

            {
                debug_time!("convert_color");
                opencv::imgproc::cvt_color(&src_frame, curr_frame, COLOR_BGR2GRAY, 0)
                    .map_err(OpticFlowRotationError::ConvertToGreyscale)?;
            }

            let mut debug_output = Mat::default();
            if let Some(debug) = &self.debug {
                debug_time!("(debug_output) resize");
                opencv::imgproc::resize(&src_frame, &mut debug_output, 
                Size2i::new(0, 0), debug.scale as f64, debug.scale as f64, INTER_AREA)?;
            }

            for attempt in 0..2 {
                let mut prev_points = Mat::default();
                let mut point_status = Mat::default();
                let mut point_errors = Mat::default();

                if retracker.should_retrack() {
                    debug_time!("good_features_to_track");
                    good_features_to_track(
                        prev_frame,
                        &mut prev_points,
                        300,
                        0.05,
                        50.0,
                        &no_array()?,
                        5,
                        false,
                        0.04,
                    ).map_err(OpticFlowRotationError::GoodFeaturesToTrack)?;
                    retracker.retracking();
                } else {
                    prev_points.clone_from(&curr_points);
                };

                {
                    debug_time!("calc_optical_flow_pyr_lk");
                    calc_optical_flow_pyr_lk(
                        prev_frame,
                        curr_frame,
                        &prev_points,
                        &mut curr_points,
                        &mut point_status,
                        &mut point_errors,
                        Size2i::new(21, 21),
                        3,
                        TermCriteria::new(
                            TermCriteria_Type::COUNT as i32 + TermCriteria_Type::EPS as i32,
                            30,
                            0.01,
                        )
                        .map_err(OpticFlowRotationError::CalculateOpticFlow)?,
                        0,
                        1e-4,
                    )
                    .map_err(OpticFlowRotationError::CalculateOpticFlow)?;
                }

                let mut points = Vec::new();
                let prev_points_vec = prev_points
                    .to_vec_2d::<Point2f>()
                    .map_err(OpticFlowRotationError::PointsDataFormat)?;
                let curr_points_vec = curr_points
                    .to_vec_2d::<Point2f>()
                    .map_err(OpticFlowRotationError::PointsDataFormat)?;
                let point_status = point_status
                    .to_vec_2d::<u8>()
                    .map_err(OpticFlowRotationError::PointsDataFormat)?;
                let point_errors = point_errors
                    .to_vec_2d::<f32>()
                    .map_err(OpticFlowRotationError::PointsDataFormat)?;

                for (prev_point, curr_point, status, error) in
                    izip!(prev_points_vec, curr_points_vec, point_status, point_errors)
                {
                    assert_eq!(prev_point.len(), 1);
                    assert_eq!(curr_point.len(), 1);
                    assert_eq!(status.len(), 1);
                    assert_eq!(error.len(), 1);

                    if status[0] != 1 {
                        continue;
                    }

                    let prev = Point2::new(prev_point[0].x, prev_point[0].y);
                    let curr = Point2::new(curr_point[0].x, curr_point[0].y);
                    if prev.coords[0] < min_x || prev.coords[0] > max_x {
                        continue;
                    }
                    if prev.coords[1] < min_y || prev.coords[1] > max_y {
                        continue;
                    }
                    if curr.coords[0] < min_x || curr.coords[0] > max_x {
                        continue;
                    }
                    if curr.coords[1] < min_y || curr.coords[1] > max_y {
                        continue;
                    }

                    if let Some(debug) = &self.debug {
                        opencv::imgproc::line(&mut debug_output, 
                            Point2i::new((prev.coords.x * debug.scale) as i32, (prev.coords.y * debug.scale) as i32),
                            Point2i::new((curr.coords.x * debug.scale) as i32, (curr.coords.y * debug.scale) as i32),
                            Scalar::new(0.0, 0.0, 255.0, 255.0),
                            3 as i32, LINE_8, 0)?;
                    }

                    let point = OpticFlowPoint {
                        prev,
                        curr,
                        error: error[0],
                    };
                    points.push(point);
                }

                let frame_optic_flow = FrameOpticFlow::new(points);

                retracker.got_optic_flow(&frame_optic_flow);
                if retracker.should_retrack() {
                    assert_eq!(attempt, 0);
                    continue;
                }

                let pose = {
                    debug_time!("recover_rotation");
                    if let Ok(pose) = self.lens.recover_rotation(&frame_optic_flow).map_err(OpticFlowRotationError::Camera) {
                        pose
                    } else {
                        let frame_rotation = FrameRotation {
                            roll: 0.0,
                            pitch: 0.0,
                            yaw: 0.0,
                            time,
                            quality: 0.0,
                        };
                        self.cache.set(ix, Json(frame_rotation)).unwrap();
                        self.frames[ix] = Some(frame_rotation);
        
                        retracker.force_retrack_next();
                        continue 'next_frame;
                    }
                };
                retracker.got_pose(&pose);
                if retracker.should_retrack() {
                    assert_eq!(attempt, 0);
                    continue;
                }

                // let scaled_rotation = UnitQuaternion::default().slerp(&pose.rotation, frame_duration * 30.0);
                // trace!("slerping with coeff {}", frame_duration * 30.0);
                let scaled_rotation = pose.rotation;
                let (rot_x, rot_y, rot_z) = scaled_rotation.euler_angles();
                let roll = -rot_z / frame_duration;
                let pitch = rot_x / frame_duration;
                let yaw = rot_y / frame_duration;

                let frame_rotation = FrameRotation {
                    roll,
                    pitch,
                    yaw,
                    time,
                    quality: retracker.current_quality,
                };
                self.cache.set(ix, Json(frame_rotation)).unwrap();
                self.frames[ix] = Some(frame_rotation);

                if let Some(_) = &self.debug {
                    debug_time!("(debug_output) undistort_image + imshow");
                    let mut undistorted = self.lens.undistort_image(&debug_output).unwrap();
                    opencv::imgproc::put_text(&mut undistorted, 
                        &format!("R: {:+03.2}deg/s", roll * 180.0 / PI), 
                        Point2i::new(0, 100), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::new(255.0, 0.0, 0.0, 255.0),
                        2, LINE_AA, false)?;
                    opencv::imgproc::put_text(&mut undistorted, 
                        &format!("P: {:+03.2}deg/s", pitch * 180.0 / PI), 
                        Point2i::new(0, 150), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::new(255.0, 0.0, 0.0, 255.0),
                        2, LINE_AA, false)?;
                    opencv::imgproc::put_text(&mut undistorted, 
                        &format!("Y: {:+03.2}deg/s", yaw * 180.0 / PI),
                        Point2i::new(0, 200), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::new(255.0, 0.0, 0.0, 255.0),
                        2, LINE_AA, false)?;
                    imshow("frame", &undistorted)?;
                }


                break;
            }
        }
        Ok(())
    }

    pub fn get_by_ix(&mut self, ix: usize) -> Option<FrameRotation> {
        self.frames.get(ix).and_then(|s| s.as_ref().copied())
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }
}

#[derive(Clone)]
pub struct OpticFlowData {
    pub timestep: f64,
    pub time: Vec<f64>,
    pub pitch: Vec<f64>,
    pub yaw: Vec<f64>,
    pub roll: Vec<f64>,
}

impl OpticFlowData {
    #[allow(unused)]
    pub fn filtered(&self, window_size: usize) -> OpticFlowData {
        OpticFlowData {
            timestep: self.timestep,
            time: self.time[window_size-1..].iter().copied().collect(),
            pitch: sliding_average(&self.pitch, window_size),
            yaw: sliding_average(&self.yaw, window_size),
            roll: sliding_average(&self.roll, window_size),
        }
    }

    pub fn start_time(&self) -> f64 {
        self.time[0]
    }
}

pub fn read_optic_flow_data(video: &mut OpticFlowRotation, timestep: f64, start_frame: usize, frames: usize) -> OpticFlowData {
    let frames = frames.min(video.len() - start_frame);

    video.recalc_for_ix_range_if_needed(start_frame..start_frame+frames).unwrap();
    let mut last_frame = video.get_by_ix(start_frame).unwrap();

    let mut of_time = Vec::new();
    let mut of_pitch = Vec::new();
    let mut of_yaw = Vec::new();
    let mut of_roll = Vec::new();
    of_time.push(last_frame.time);
    of_pitch.push(last_frame.pitch);
    of_yaw.push(last_frame.yaw);
    of_roll.push(last_frame.roll);

    let mut timestep_carry = 0.0;
    
    for ix in start_frame+1..(start_frame + frames) {
        let mut curr_frame = video.get_by_ix(ix).expect(&format!("video frame {}", ix));

        let actual_timestep = curr_frame.time - last_frame.time;
        // 8 = 8.325 = 0.0333 / 0.004
        let subframes = ((actual_timestep + timestep_carry) / timestep).round();
        // 0 + 0.0333 - 8 * 0.004 = 0.0333 - 0.032 = 0.0013
        timestep_carry += actual_timestep - subframes * timestep;
        // 9 = 8.65 = (0.0333 + 0.0013) / 0.004
        // 0.0013 + 0.0333 - 9 * 0.004 = -0.0014

        let subframes = subframes as usize;

        let mut fake_time = *of_time.last().unwrap();
        let mut fake_pitch = *of_pitch.last().unwrap();
        let mut fake_yaw = *of_yaw.last().unwrap();
        let mut fake_roll = *of_roll.last().unwrap();

        let time_slope = (curr_frame.time - fake_time) / subframes as f64;
        let pitch_slope = (curr_frame.pitch - fake_pitch) / subframes as f64;
        let yaw_slope = (curr_frame.yaw - fake_yaw) / subframes as f64;
        let roll_slope = (curr_frame.roll - fake_roll) / subframes as f64;

        for _ in 0..subframes {
            fake_time += time_slope;
            fake_pitch += pitch_slope;
            fake_yaw += yaw_slope;
            fake_roll += roll_slope;

            of_time.push(fake_time);
            of_pitch.push(fake_pitch);
            of_yaw.push(fake_yaw);
            of_roll.push(fake_roll);
        }

        std::mem::swap(&mut last_frame, &mut curr_frame);
    }

    OpticFlowData {
        timestep,
        time: of_time,
        pitch: of_pitch,
        yaw: of_yaw,
        roll: of_roll,
    }
}

