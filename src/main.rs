use std::{
    f64::consts::PI,
    fs::File,
    io::{BufWriter, Read, Write},
    path::PathBuf,
    str::FromStr,
    sync::{Arc, Mutex},
};

use clap::Clap;
use fc_blackbox::BlackboxRecord;

use itertools::Itertools;
use itertools_num::linspace;
use kv::{Config, Store};
use measure_time::*;
use nalgebra::{Quaternion, Unit, UnitQuaternion};
use opencv::videoio::VideoCapture;
use opts::{Opts, SubCommand};
use rayon::prelude::*;

use crate::{
    bbox::read_bbox_data,
    camera::Lens,
    offset::{
        find_offset_at_frame, find_offset_at_frame_full_bbox, offset_inliers, EstimatedOffset,
        InterpolatingOffsetMapper, LinearOffsetMapper,
    },
    optic_flow::{DebugOptions, OpticFlowRotation},
    video_source::VideoSource,
};

mod bbox;
mod camera;
mod offset;
mod optic_flow;
mod opts;
mod util;
mod video_source;

fn main() -> Result<(), anyhow::Error> {
    pretty_env_logger::init();
    let opts = Opts::parse();

    let debug = opts.debug_output.map(|scale| DebugOptions { scale });
    if debug.is_some() {
        opencv::highgui::start_window_thread()?;
    }

    match opts.subcmd {
        SubCommand::DumpOpticFlow(dump_csv) => {
            let camera = Lens::from_file(&dump_csv.camera)?;

            let video_path = PathBuf::from_str(&dump_csv.video)?;
            let output_filename = &dump_csv.output.unwrap_or_else(|| {
                video_path
                    .with_extension("of.csv")
                    .to_string_lossy()
                    .into_owned()
            });

            let cap = VideoCapture::from_file(
                video_path.to_string_lossy().as_ref(),
                opencv::videoio::CAP_ANY,
            )?;
            let cache = Store::new(Config::new(format!(
                "{}.cache",
                video_path.to_string_lossy()
            )))?;

            let mut frame_rotation = OpticFlowRotation::new(
                VideoSource::new(cap)?,
                camera,
                debug,
                cache.bucket(Some("optic-flow"))?,
            );
            let range = 0..frame_rotation.len();
            if !dump_csv.cached_only {
                frame_rotation.recalc_for_ix_range_if_needed(range.clone())?;
            }

            let mut out_of = BufWriter::new(File::create(output_filename)?);
            writeln!(out_of, "time,x,y,z,quality")?;

            for frame in range {
                if let Some(rotation) = frame_rotation.get_by_ix(frame) {
                    writeln!(
                        out_of,
                        "{},{},{},{},{}",
                        rotation.time,
                        rotation.pitch, // pitch down
                        rotation.yaw,   // yaw left
                        rotation.roll,  // roll right
                        rotation.quality,
                    )?;
                }
            }
        }
        SubCommand::DumpGyro(dump_csv) => {
            let bbox_path = PathBuf::from_str(&dump_csv.bbox)?;
            let output_filename = &dump_csv.output.unwrap_or_else(|| {
                bbox_path
                    .with_extension("att.csv")
                    .to_string_lossy()
                    .into_owned()
            });

            let mut bytes = Vec::new();
            File::open(&bbox_path)?.read_to_end(&mut bytes)?;

            let mut bbox = fc_blackbox::BlackboxReader::from_bytes(&bytes)?;

            let mut out_att = BufWriter::new(File::create(output_filename)?);

            writeln!(out_att, "time,x,y,z")?;

            let time_ix = bbox.header.ip_fields["time"].ix;
            let roll_right_ix = bbox.header.ip_fields["gyroADC[0]"].ix;
            let pitch_down_ix = bbox.header.ip_fields["gyroADC[1]"].ix;
            let yaw_left_ix = bbox.header.ip_fields["gyroADC[2]"].ix;

            let gyro_scale = bbox.header.gyro_scale as f64 * 1_000_000.0; // TODO

            while let Some(record) = bbox.next() {
                match record {
                    BlackboxRecord::Main(values) => {
                        let time = values[time_ix] as f64 / 1_000_000.0;
                        writeln!(
                            out_att,
                            "{},{},{},{}",
                            time,
                            -values[pitch_down_ix] as f64 * gyro_scale, // pitch down
                            -values[yaw_left_ix] as f64 * gyro_scale,   // yaw left
                            values[roll_right_ix] as f64 * gyro_scale,  // roll right
                        )?;
                    }
                    BlackboxRecord::Garbage(length) => {
                        warn!("Got {} bytes of garbage", length);
                    }
                    _ => {}
                }
            }
        }
        SubCommand::Match(match_cmd) => {
            info!("Reading camera correction parameters");
            let camera = Lens::from_file(&match_cmd.camera)?;

            let video_path = PathBuf::from_str(&match_cmd.video)?;
            info!("Opening cache");
            let cache = Store::new(Config::new(format!(
                "{}.cache",
                video_path.to_string_lossy()
            )))?;

            info!("Opening video file");
            let cap = VideoCapture::from_file(
                video_path.to_string_lossy().as_ref(),
                opencv::videoio::CAP_ANY,
            )?;
            let video_source = VideoSource::new(cap)?;
            let video_frame_duration = video_source.frame_duration();
            let video_length_seconds = video_source.last_timestamp();
            let video_length_frames = video_source.frame_count();
            let video_fps = video_source.fps();
            info!("Read {:.1}s of video", video_length_seconds);

            let frame_rotation = OpticFlowRotation::new(
                video_source,
                camera,
                debug,
                cache.bucket(Some("optic-flow"))?,
            );
            let bbox_path = PathBuf::from_str(&match_cmd.bbox)?;
            let frame_rotation = Arc::new(Mutex::new(frame_rotation));

            let bbox = {
                info!("Reading blackbox data");
                info_time!("Reading blackbox data");
                let mut bytes = Vec::new();
                File::open(&bbox_path)?.read_to_end(&mut bytes)?;

                let mut bbox = fc_blackbox::BlackboxReader::from_bytes(&bytes)?;
                let mut bytes_read = 0;
                for _ in 0..match_cmd.bbox_ix {
                    info!("Skipping blackbox segment");
                    while let Some(record) = bbox.next() {
                        match record {
                            BlackboxRecord::Event(fc_blackbox::frame::event::Frame::EndOfLog) => {
                                bytes_read += bbox.bytes_read();
                                println!(
                                    "next bytes: {}",
                                    String::from_utf8_lossy(&bytes[bytes_read..bytes_read + 8])
                                );
                                bbox =
                                    fc_blackbox::BlackboxReader::from_bytes(&bytes[bytes_read..])?;
                                break;
                            }
                            _ => {}
                        }
                    }
                }

                read_bbox_data(&mut bbox)
            };
            info!(
                "Read {:.1}s of blackbox data",
                bbox.end_time() - bbox.start_time()
            );

            let mut camera_angle = match_cmd.angle.unwrap_or(0.0) / 180.0 * PI;

            let offset_mapper = {
                info!("Estimating rough offset");
                info_time!("Estimating rough offset");

                let mut sample_length = 20;
                let mut samples_m1 = 4;

                let mut best_offsets = None;

                for attempt in 0..6 {
                    let min_points = (samples_m1 / 2) + 3;
                    info!(
                        "Attempt {} with {} samples {} frames each",
                        attempt + 1,
                        samples_m1 + 1,
                        sample_length
                    );
                    let video_timestamps = linspace(0.3, 0.7, samples_m1 + 1)
                        .map(|offset| offset * video_length_seconds as f64)
                        .collect_vec();

                    let results: Vec<EstimatedOffset> = (video_timestamps
                        .iter()
                        .map(|o| (o, Arc::clone(&frame_rotation)))
                        .collect_vec())
                    .par_iter_mut()
                    .map(|(video_ts, frame_rotation)| {
                        find_offset_at_frame_full_bbox(
                            frame_rotation,
                            &bbox,
                            camera_angle,
                            video_fps,
                            **video_ts,
                            sample_length,
                        )
                    })
                    .collect();

                    println!("{:#?}", &results);

                    let offsets = offset_inliers(&results, 0.3);
                    if offsets.len() >= min_points {
                        best_offsets = Some(offsets);
                        break;
                    }

                    println!("{:#?}", &offsets);

                    if attempt % 2 == 0 {
                        sample_length *= 2;
                    } else {
                        samples_m1 *= 2;
                    }
                }

                if let Some(best_offsets) = best_offsets {
                    LinearOffsetMapper::from_offsets(&best_offsets)
                } else {
                    panic!("Can't estimate rough offset");
                }
            };

            let offset_mapper = {
                info!("Refining camera angle (might take a while)");
                info_time!("Refining camera angle");

                let video_ts_at_bbox_start = offset_mapper.video_ts_at_bbox_ts(bbox.start_time());
                let video_ts_at_bbox_end = offset_mapper.video_ts_at_bbox_ts(bbox.end_time());

                let video_flight_session_length = video_ts_at_bbox_end - video_ts_at_bbox_start;

                let video_timestamps = linspace(
                    video_ts_at_bbox_start + 0.2 * video_flight_session_length,
                    video_ts_at_bbox_end - 0.2 * video_flight_session_length,
                    9,
                )
                .collect_vec();

                let fanout = 11;
                let mut angle_range = match_cmd.angle_range.unwrap_or_else(|| {
                    if match_cmd.angle.is_some() {
                        10.0
                    } else {
                        90.0
                    }
                }) / 180.0
                    * PI;
                let min_range = 1.0 / 180.0 * PI;

                let mut best_offsets = vec![];

                loop {
                    let mut best_angle = camera_angle;
                    let mut min_cost = f64::INFINITY;
                    let min_angle = camera_angle - angle_range / 2.0;
                    let max_angle = camera_angle + angle_range / 2.0;
                    info!(
                        "Searching for camera angle between {:.02}deg and {:.02}deg",
                        min_angle * 180.0 / PI,
                        max_angle * 180.0 / PI
                    );

                    for angle in linspace(min_angle, max_angle, fanout) {
                        let offsets: Vec<_> = (video_timestamps
                            .iter()
                            .map(|o| (o, Arc::clone(&frame_rotation)))
                            .collect_vec())
                        .par_iter_mut()
                        .map(|(video_ts, frame_rotation)| {
                            find_offset_at_frame(
                                frame_rotation,
                                &bbox,
                                angle,
                                video_fps,
                                **video_ts,
                                60,
                                offset_mapper.bbox_ts_at_video_ts(**video_ts),
                                10.0,
                            )
                        })
                        .collect();

                        let offsets = offset_inliers(&offsets, 0.5);

                        let avg_cost =
                            offsets.iter().map(|o| o.cost).sum::<f64>() / offsets.len() as f64;
                        let penalty = 1.5f64.powi((video_timestamps.len() - offsets.len()) as i32);
                        let cost = avg_cost * penalty;

                        if cost < min_cost {
                            min_cost = cost;
                            best_angle = angle;
                            best_offsets = offsets;
                        }
                    }

                    camera_angle = best_angle;

                    if angle_range < min_range {
                        break;
                    }
                    angle_range *= 3.0 / fanout as f64;
                }

                info!("Found camera angle: {:.02}deg", camera_angle * 180.0 / PI);

                LinearOffsetMapper::from_offsets(&best_offsets)
            };

            let video_ts_at_bbox_start = offset_mapper.video_ts_at_bbox_ts(bbox.start_time());
            let video_ts_at_bbox_end = offset_mapper.video_ts_at_bbox_ts(bbox.end_time());

            let video_flight_session_length = video_ts_at_bbox_end - video_ts_at_bbox_start;

            let video_timestamps = linspace(
                video_ts_at_bbox_start + 0.2 * video_flight_session_length,
                video_ts_at_bbox_end - 0.2 * video_flight_session_length,
                17,
            )
            .collect_vec();

            let offset_mapper = {
                info!("Refining offsets across the whole video");
                info_time!("Refining offsets across the whole video");

                let results: Vec<_> = (video_timestamps
                    .iter()
                    .map(|o| (o, Arc::clone(&frame_rotation)))
                    .collect_vec())
                .par_iter()
                .map(|(video_ts, frame_rotation)| {
                    find_offset_at_frame(
                        frame_rotation,
                        &bbox,
                        camera_angle,
                        video_fps,
                        **video_ts,
                        120,
                        offset_mapper.bbox_ts_at_video_ts(**video_ts),
                        10.0,
                    )
                })
                .collect();

                let results = offset_inliers(&results, 0.5);

                InterpolatingOffsetMapper::from_offsets(&results)
            };

            {
                info!("Exporting Davinci Resolve splines");
                info_time!("Exporting Davinci Resolve splines");

                // roll around 30846.0

                let tau = match_cmd.tau.unwrap_or(0.2);
                let frame_center_offset =
                    match_cmd.frame_center_offset.unwrap_or(-0.5) * video_frame_duration;

                let mut attitude = AttitudeEstimator::new();
                let alpha = 1.0 - (-video_frame_duration / tau).exp();
                let mut smoothed_attitude = SlerpSmoother::new(attitude.attitude(), alpha);
                let mut bbox_frame_start =
                    bbox.ix_at(offset_mapper.bbox_ts_at_video_ts(0.0 + frame_center_offset));

                let output_prefix = match_cmd.output.unwrap_or_else(|| {
                    video_path.with_extension("").to_string_lossy().into_owned()
                });
                let mut out_w = BufWriter::new(File::create(format!("{}.w.spl", output_prefix))?);
                let mut out_x = BufWriter::new(File::create(format!("{}.x.spl", output_prefix))?);
                let mut out_y = BufWriter::new(File::create(format!("{}.y.spl", output_prefix))?);
                let mut out_z = BufWriter::new(File::create(format!("{}.z.spl", output_prefix))?);

                writeln!(out_w, "DFSP").unwrap();
                writeln!(out_x, "DFSP").unwrap();
                writeln!(out_y, "DFSP").unwrap();
                writeln!(out_z, "DFSP").unwrap();

                for frame_ix in 0..video_length_frames {
                    let frame_ts =
                        (frame_ix as f64 + 1.0) * video_frame_duration + frame_center_offset;
                    let range_to_apply = {
                        let bbox_frame_end =
                            bbox.ix_at(offset_mapper.bbox_ts_at_video_ts(frame_ts));
                        let bbox_frame_range = match (bbox_frame_start, bbox_frame_end) {
                            (Some(begin), Some(end)) => begin..end,
                            (None, Some(end)) => 0..end,
                            (Some(begin), None) => begin..bbox.len(),
                            (None, None) => 0..0,
                        };
                        bbox_frame_start = bbox_frame_end;
                        bbox_frame_range
                    };

                    for ix in range_to_apply {
                        attitude.apply_euler_angles(
                            bbox.pitch[ix],
                            bbox.yaw[ix],
                            bbox.roll[ix],
                            bbox.timestep,
                        );
                    }
                    smoothed_attitude.apply_attitude(attitude.attitude());
                    let correction = smoothed_attitude.correction_from(attitude.attitude());

                    writeln!(out_w, "{} {}", frame_ix, correction.coords.w).unwrap();
                    writeln!(out_x, "{} {}", frame_ix, correction.coords.x).unwrap();
                    writeln!(out_y, "{} {}", frame_ix, correction.coords.y).unwrap();
                    writeln!(out_z, "{} {}", frame_ix, correction.coords.z).unwrap();
                }
            }
        }
    }

    Ok(())
}

struct AttitudeEstimator {
    attitude: Unit<Quaternion<f64>>,
}

impl AttitudeEstimator {
    pub fn new() -> Self {
        Self {
            attitude: UnitQuaternion::identity(),
        }
    }

    pub fn apply_euler_angles(&mut self, roll: f64, pitch: f64, yaw: f64, scale: f64) {
        self.attitude = UnitQuaternion::identity()
            .slerp(&UnitQuaternion::from_euler_angles(roll, pitch, yaw), scale)
            * self.attitude;
    }

    pub fn attitude(&self) -> Unit<Quaternion<f64>> {
        self.attitude
    }
}

struct SlerpSmoother {
    alpha: f64,
    attitude: Unit<Quaternion<f64>>,
}

impl SlerpSmoother {
    pub fn new(attitude: Unit<Quaternion<f64>>, alpha: f64) -> Self {
        Self { attitude, alpha }
    }

    pub fn apply_attitude(&mut self, attitude: Unit<Quaternion<f64>>) {
        self.attitude = self.attitude.slerp(&attitude, self.alpha);
    }

    pub fn correction_from(&self, attitude: Unit<Quaternion<f64>>) -> Unit<Quaternion<f64>> {
        attitude.rotation_to(&self.attitude)
    }
}
