use fc_blackbox::{BlackboxReader, BlackboxRecord};
use itertools::{izip, Itertools};
use nalgebra::{Rotation, Vector3};
use rustfft::{
    num_complex::{Complex, Complex32},
    FftPlanner,
};

use crate::util::sliding_average;

#[derive(Clone)]
pub struct GyroData {
    pub timestep: f64,
    pub time: Vec<f64>,
    pub pitch: Vec<f64>,
    pub yaw: Vec<f64>,
    pub roll: Vec<f64>,
}

impl GyroData {
    #[allow(unused)]
    pub fn filtered(&self, window_size: usize) -> GyroData {
        GyroData {
            timestep: self.timestep,
            time: self.time[window_size - 1..].iter().copied().collect(),
            pitch: sliding_average(&self.pitch, window_size),
            yaw: sliding_average(&self.yaw, window_size),
            roll: sliding_average(&self.roll, window_size),
        }
    }

    pub fn subset(&self, start_time: f64, length: f64) -> GyroData {
        let start_ix = self.ix_at(start_time).unwrap();
        let end_ix = (start_ix + (length / self.timestep).round() as usize) + 1;
        GyroData {
            timestep: self.timestep,
            time: self.time[start_ix..end_ix].iter().copied().collect(),
            pitch: self.pitch[start_ix..end_ix].iter().copied().collect(),
            yaw: self.yaw[start_ix..end_ix].iter().copied().collect(),
            roll: self.roll[start_ix..end_ix].iter().copied().collect(),
        }
    }

    pub fn with_camera_angle(&self, angle: f64) -> GyroData {
        let adjust_before = Rotation::from_axis_angle(&Vector3::x_axis(), -angle); // pitch up is positive
        let adjust_after = adjust_before.inverse();

        let mut roll_adjusted = Vec::new();
        let mut pitch_adjusted = Vec::new();
        let mut yaw_adjusted = Vec::new();
        for (&roll, &pitch, &yaw) in izip!(&self.roll, &self.pitch, &self.yaw) {
            let rotation = Rotation::from_euler_angles(roll, pitch, yaw);
            let (new_roll, new_pitch, new_yaw) =
                (adjust_after * rotation * adjust_before).euler_angles();
            roll_adjusted.push(new_roll);
            pitch_adjusted.push(new_pitch);
            yaw_adjusted.push(new_yaw);
        }

        GyroData {
            timestep: self.timestep,
            time: self.time.clone(),
            pitch: pitch_adjusted,
            yaw: yaw_adjusted,
            roll: roll_adjusted,
        }
    }

    pub fn start_time(&self) -> f64 {
        self.time[0]
    }

    pub fn end_time(&self) -> f64 {
        self.time[self.time.len() - 1]
    }

    pub fn ix_at(&self, time: f64) -> Option<usize> {
        let guess = (time - self.start_time()) / self.timestep;
        if guess < 0.0 {
            None
        } else if guess > self.time.len() as f64 {
            None
        } else {
            Some(guess as usize)
        }
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn good_angle_estimation_fragments(&self) -> Vec<f64> {
        let loop_time = (1.0 / self.timestep).round() as usize;
        assert!(loop_time >= 64);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(loop_time);

        let mut buffer = vec![
            Complex {
                re: 0.0f64,
                im: 0.0f64
            };
            loop_time
        ];
        let mut scores = vec![];

        for second in 0..self.end_time().floor() as usize {
            for ix in 0..loop_time {
                buffer[ix] = Complex {
                    re: self.roll[second * loop_time + ix],
                    im: 0.0,
                };
            }

            fft.process(&mut buffer);

            // 0..1 Hz ignore
            // 1..(fps/2) region of interest
            // (fps/2)..fps better be quiet
            // fps.. ignore
            // f_s = loop_time
            // bin_width = 1Hz

            let mut score = 0.0;
            for freq in 1..16 {
                score += buffer[freq].re;
            }
            for freq in 16..31 {
                score -= buffer[freq].re * 0.5;
            }
            scores.push(score);
        }

        unimplemented!();
    }
}

fn find_peak(data: &mut [f64], cutout_width: f64) -> (usize, f64) {
    let max_ix = data
        .iter()
        .position_max_by(|a, b| PartialOrd::partial_cmp(*a, *b).unwrap())
        .unwrap();
    let max = data[max_ix];
    // let
    for ix in (max_ix - cutout_width as usize)..(max_ix - cutout_width as usize) {
        let distance = (ix as isize - max_ix as isize) as f64;
        // let
    }

    unimplemented!()
}

pub fn read_bbox_data(bbox: &mut BlackboxReader) -> GyroData {
    // let time_ix = bbox.header.ip_fields["time"].ix;
    let loop_iteration_ix = bbox.header.ip_fields["loopIteration"].ix;
    let roll_right_ix = bbox.header.ip_fields["gyroADC[0]"].ix;
    let pitch_down_ix = bbox.header.ip_fields["gyroADC[1]"].ix;
    let yaw_left_ix = bbox.header.ip_fields["gyroADC[2]"].ix;
    let loop_time: u32 = bbox.header.loop_time;
    let pid_process_denom: u32 = bbox.header.other_headers["pid_process_denom"]
        .parse()
        .unwrap();
    let time_scale = (loop_time * pid_process_denom) as f64 / 1_000_000.0;

    let gyro_scale = bbox.header.gyro_scale as f64 * 1_000_000.0; // TODO

    let mut gyro_time = Vec::new();
    let mut gyro_pitch = Vec::new();
    let mut gyro_yaw = Vec::new();
    let mut gyro_roll = Vec::new();

    // let mut time_base = 0u64;
    // let mut last_raw_time = None;
    let mut last_time = None;
    let mut valid_intervals_sum = 0.0;
    let mut valid_intervals_qty = 0usize;

    'read: while let Some(record) = bbox.next() {
        match record {
            BlackboxRecord::Main(values) => {
                // let raw_time = values[time_ix] as i32 as u32;
                // if let Some(last_raw_time) = last_raw_time {
                //     if raw_time < last_raw_time {
                //         time_base += 0x1_0000_0000u64;
                //     }
                // }

                // let time = (raw_time as u64 + time_base) as f64 / 1_000_000.0;
                let time = values[loop_iteration_ix] as f64 * time_scale;
                let pitch = values[pitch_down_ix] as f64 * gyro_scale;
                let yaw = values[yaw_left_ix] as f64 * gyro_scale;
                let roll = values[roll_right_ix] as f64 * gyro_scale;

                let valid_interval = if valid_intervals_qty > 0 {
                    let expected_interval = valid_intervals_sum / valid_intervals_qty as f64;
                    let this_interval: f64 = time - last_time.unwrap();
                    if this_interval > expected_interval * 1.55 {
                        let missed_frames =
                            (this_interval / expected_interval).round() as usize - 1;

                        let mut fake_time = last_time.unwrap();
                        let mut fake_pitch = *gyro_pitch.last().unwrap();
                        let mut fake_yaw = *gyro_yaw.last().unwrap();
                        let mut fake_roll = *gyro_roll.last().unwrap();

                        let time_slope = (time - fake_time) / missed_frames as f64;
                        let pitch_slope = (pitch - fake_pitch) / missed_frames as f64;
                        let yaw_slope = (yaw - fake_yaw) / missed_frames as f64;
                        let roll_slope = (roll - fake_roll) / missed_frames as f64;

                        for _ in 0..missed_frames {
                            fake_time += time_slope;
                            fake_pitch += pitch_slope;
                            fake_yaw += yaw_slope;
                            fake_roll += roll_slope;

                            gyro_time.push(fake_time);
                            gyro_pitch.push(fake_pitch);
                            gyro_yaw.push(fake_yaw);
                            gyro_roll.push(fake_roll);
                        }
                        false
                    } else {
                        true
                    }
                } else {
                    true
                };

                gyro_time.push(time);
                gyro_pitch.push(pitch);
                gyro_yaw.push(yaw);
                gyro_roll.push(roll);

                if valid_interval {
                    if let Some(last_time) = last_time {
                        valid_intervals_sum += time - last_time;
                        valid_intervals_qty += 1;
                    }
                }
                last_time = Some(time);
                // last_raw_time = Some(raw_time);
            }
            BlackboxRecord::Garbage(length) => {
                println!("Got {} bytes of garbage", length);
            }
            BlackboxRecord::Event(fc_blackbox::frame::event::Frame::EndOfLog) => {
                break 'read;
            }
            _ => {}
        }
    }

    GyroData {
        timestep: valid_intervals_sum / valid_intervals_qty as f64,
        time: gyro_time,
        pitch: gyro_pitch,
        yaw: gyro_yaw,
        roll: gyro_roll,
    }
}
