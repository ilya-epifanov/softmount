use std::{
    fmt::{Debug, Display},
    sync::{Arc, Mutex},
};

use itertools::Itertools;
use linreg::linear_regression_of;
use measure_time::*;

use crate::{
    bbox::GyroData,
    optic_flow::{read_optic_flow_data, OpticFlowData, OpticFlowRotation},
    util::ransac,
};

#[derive(Default, Clone, Copy)]
pub struct EstimatedOffset {
    pub at_video_ts: f64,
    pub offset: f64,
    pub cost: f64,
}

impl Display for EstimatedOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} at {} with cost of {:.04}",
            self.offset, self.at_video_ts, self.cost
        )
    }
}

impl Debug for EstimatedOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self, f)
    }
}

impl EstimatedOffset {
    pub fn offset_at_video_ts(&self) -> (f64, f64) {
        (self.at_video_ts, self.offset)
    }
}

pub struct LinearOffsetMapper {
    k: f64,
    b: f64,
}

impl LinearOffsetMapper {
    pub fn from_offsets(offsets: &[EstimatedOffset]) -> Self {
        let (k, b): (f64, f64) = linreg::linear_regression_of(
            &offsets.iter().map(|o| o.offset_at_video_ts()).collect_vec(),
        )
        .unwrap();
        Self { k, b }
    }

    pub fn bbox_ts_at_video_ts(&self, video_ts: f64) -> f64 {
        video_ts + video_ts * self.k + self.b
    }

    pub fn video_ts_at_bbox_ts(&self, bbox_ts: f64) -> f64 {
        // B = V*(1+k)+b
        // V = (B-b)/(1+k)
        (bbox_ts - self.b) / (1.0 + self.k)
    }
}

pub struct InterpolatingOffsetMapper {
    offsets: Vec<(f64, f64)>,
}

impl InterpolatingOffsetMapper {
    pub fn from_offsets(offsets: &[EstimatedOffset]) -> Self {
        Self {
            offsets: offsets.iter().map(|o| o.offset_at_video_ts()).collect_vec(),
        }
    }

    fn offset_at_video_ts(&self, video_ts: f64) -> f64 {
        let last_time = self.offsets[0].0 - (self.offsets[1].0 - self.offsets[0].0);
        let last_offset = self.offsets[0].1 - (self.offsets[1].1 - self.offsets[0].1);
        for (vt, offset) in self.offsets.iter().copied() {
            if video_ts < vt {
                let a = (video_ts - last_time) / (vt - last_time);
                return a * offset + (1.0 - a) * last_offset;
            }
        }

        let time_2 = self.offsets[self.offsets.len() - 2].0;
        let time_1 = self.offsets[self.offsets.len() - 1].0;
        let a = (video_ts - time_2) / (time_1 - time_2);
        let offset_2 = self.offsets[self.offsets.len() - 2].1;
        let offset_1 = self.offsets[self.offsets.len() - 1].1;
        a * offset_1 + (1.0 - a) * offset_2
    }

    pub fn bbox_ts_at_video_ts(&self, video_ts: f64) -> f64 {
        video_ts + self.offset_at_video_ts(video_ts)
    }
}

pub fn offset_inliers(offsets: &[EstimatedOffset], inlier_prob: f64) -> Vec<EstimatedOffset> {
    let mask = ransac(
        &offsets,
        inlier_prob,
        0.9999,
        |model: [EstimatedOffset; 2], vs: &[EstimatedOffset], mask: &mut [bool]| {
            let (k, b): (f64, f64) = linear_regression_of(&[
                (model[0].at_video_ts, model[0].offset),
                (model[1].at_video_ts, model[1].offset),
            ])
            .unwrap();

            let x_center = (model[0].at_video_ts + model[1].at_video_ts) / 2.0;
            const MAX_OFFSET_DEVIATION_PER_SECOND: f64 = 0.001;
            let mut score = 0.0;
            for (ix, offset) in vs.iter().enumerate() {
                let expected_y = k * offset.at_video_ts + b;
                let error = expected_y - offset.offset;
                if error.abs()
                    < (offset.at_video_ts - x_center).abs() * MAX_OFFSET_DEVIATION_PER_SECOND
                {
                    score += 1.0;
                    mask[ix] = true;
                }
            }

            score
        },
    );

    offsets
        .iter()
        .copied()
        .zip(mask.iter().copied())
        .filter_map(|(o, inlier)| if inlier { Some(o) } else { None })
        .collect()
}

pub fn find_offset_at_frame_full_bbox<'a, 'b>(
    frame_rotation: &Arc<Mutex<OpticFlowRotation>>,
    bbox: &'b GyroData,
    angle: f64,
    fps: f64,
    video_ts: f64,
    frames: usize,
) -> EstimatedOffset {
    let center_frame = (video_ts * fps).round() as usize;
    let of = {
        debug_time!("read_optic_flow_data");
        read_optic_flow_data(
            &mut frame_rotation.lock().unwrap(),
            bbox.timestep,
            center_frame - frames / 2,
            frames,
        )
    };

    let bbox = bbox.with_camera_angle(angle);

    let (offset_s, cost) = {
        debug_time!("find_offset");
        find_offset(&of, &bbox)
    };

    EstimatedOffset {
        at_video_ts: video_ts,
        offset: offset_s,
        cost,
    }
}

pub fn find_offset_at_frame<'a, 'b>(
    frame_rotation: &Arc<Mutex<OpticFlowRotation>>,
    bbox: &'b GyroData,
    angle: f64,
    fps: f64,
    video_ts: f64,
    frames: usize,
    center_bbox_ts: f64,
    bbox_length: f64,
) -> EstimatedOffset {
    let center_frame = (video_ts * fps).round() as usize;
    let of = {
        debug_time!("read_optic_flow_data");
        read_optic_flow_data(
            &mut frame_rotation.lock().unwrap(),
            bbox.timestep,
            center_frame - frames / 2,
            frames,
        )
    };

    let bbox = bbox
        .subset(center_bbox_ts - bbox_length / 2.0, bbox_length)
        .with_camera_angle(angle);

    let (offset_s, cost) = {
        debug_time!("find_offset");
        find_offset(&of, &bbox)
    };

    EstimatedOffset {
        at_video_ts: video_ts,
        offset: offset_s,
        cost,
    }
}

fn cross_similarity_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    let mut sqr_sum = 0.0;

    for ix in 0..a.len() {
        sqr_sum += (a[ix] - b[ix]).powi(2).min(1.0);
    }

    sqr_sum.sqrt()
}

fn find_offset(optic_flow: &OpticFlowData, gyro: &GyroData) -> (f64, f64) {
    assert!((optic_flow.timestep - gyro.timestep).abs() < 0.0001);

    let window_size = optic_flow.pitch.len();
    let scan_size = gyro.pitch.len() - window_size + 1;

    let mut min_cost = f64::INFINITY;
    let mut best_offset = 0;
    for offset in 0..scan_size {
        let cost = 0.0
            + cross_similarity_error(&gyro.pitch[offset..offset + window_size], &optic_flow.pitch)
            + cross_similarity_error(&gyro.yaw[offset..offset + window_size], &optic_flow.yaw)
            + cross_similarity_error(&gyro.roll[offset..offset + window_size], &optic_flow.roll)
                * 2.0;

        if cost < min_cost {
            min_cost = cost;
            best_offset = offset;
        }
    }

    (
        gyro.time[best_offset] - optic_flow.start_time(),
        min_cost / window_size as f64,
    )
}
