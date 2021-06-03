use std::{fs::File, path::Path};

use anyhow::*;
use log::trace;
use nalgebra::Rotation;
use serde::{Deserialize, Deserializer};
use opencv::{calib3d::{RANSAC, find_essential_mat_matrix, fisheye_undistort_points, recover_pose_camera}, core::{CV_32F, no_array}, prelude::{Mat, MatExprTrait, MatTrait, MatTraitManual}};
use thiserror::Error;

use crate::{optic_flow::{FrameOpticFlow, FramePose}, util::opencv_to_matrix3x3};

use self::json::LensJson;

#[derive(Clone, Copy, Deserialize)]
pub struct Dimensions {
    pub w: usize,
    pub h: usize,
}

#[derive(Clone)]
pub struct Lens {
    camera_matrix: Mat,
    distortion_coeffs: Mat,
    pub dimensions: Dimensions,
}

impl <'de> Deserialize<'de> for Lens {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de> {
        let lens = LensJson::deserialize(deserializer)?;
        Ok(Lens {
            camera_matrix: lens.fisheye_params.camera_matrix,
            distortion_coeffs: lens.fisheye_params.distortion_coeffs,
            dimensions: lens.calib_dimension,
        })
    }
}

mod json {
    use opencv::prelude::{Mat, MatTrait};
    use serde::*;

    use super::Dimensions;

    #[derive(Deserialize)]
    pub(crate) struct LensJson {
        pub calib_dimension: Dimensions,
        pub fisheye_params: LensParams,
    }
    
    #[derive(Deserialize)]
    pub struct LensParams {
        #[serde(deserialize_with = "deserialize_camera_matrix")]
        pub camera_matrix: Mat,
        #[serde(deserialize_with = "deserialize_distortion_coeffs")]
        pub distortion_coeffs: Mat,
    }

    pub fn deserialize_camera_matrix<'de, D>(deserializer: D) -> Result<Mat, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mat: Vec<Vec<f64>> = Deserialize::deserialize(deserializer)?;
        let mat = Mat::from_slice_2d(&mat).unwrap();
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 3);
        Ok(mat)
    }

    pub fn deserialize_distortion_coeffs<'de, D>(deserializer: D) -> Result<Mat, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mat: Vec<f64> = Deserialize::deserialize(deserializer)?;
        let mat = Mat::from_slice(&mat).unwrap();
        assert_eq!(mat.cols(), 4);
        Ok(mat)
    }
}


#[derive(Debug, Error)]
pub enum CameraError {
    #[error("couldn't undistort points, {0}")]
    UndistortPoints(opencv::Error),
    #[error("opencv error, {0}")]
    OpenCV(#[from] opencv::Error),
}

#[derive(Debug, Error)]
pub enum CameraDeserializationError {
    #[error("coulnd't read camera parameters file")]
    IOError(#[from] std::io::Error),
    #[error("coulnd't parse camera parameters file")]
    DeserializationError(#[from] serde_json::Error),
}

impl Lens {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, CameraDeserializationError> {
        let reader = File::open(path)?;
        let camera: Lens = serde_json::from_reader(reader)?;
        Ok(camera)
    }

    fn undistort_points(&self, points: &Mat) -> Result<Mat, CameraError> {
        let mut undistorted_points = points.clone();
        fisheye_undistort_points(&points, &mut undistorted_points, &self.camera_matrix, &self.distortion_coeffs, &no_array()?, &self.camera_matrix).map_err(CameraError::UndistortPoints)?;
        Ok(undistorted_points)
    }

    pub fn undistort_image(&self, image: &Mat) -> Result<Mat, CameraError> {
        let scale = image.rows() as f64 / self.dimensions.h as f64;
        let size = opencv::core::Size_::new(image.cols(), image.rows());

        let mut camera_matrix = self.camera_matrix.clone();
        for i in 0..3 {
            for j in 0..3 {
                *camera_matrix.at_2d_mut::<f64>(i, j)? *= scale;
            }
        }
        *camera_matrix.at_2d_mut::<f64>(2, 2)? = 1.0;

        let mut new_k = Mat::default();
        opencv::calib3d::estimate_new_camera_matrix_for_undistort_rectify(
            &camera_matrix, 
            &self.distortion_coeffs, 
            size,
            &Mat::eye(3, 3, opencv::core::CV_32F).unwrap(),
            &mut new_k,
            0.0,
            size,
            1.1,
        ).unwrap();

        let mut map1 = Mat::default();
        let mut map2 = Mat::default();
        opencv::calib3d::fisheye_init_undistort_rectify_map(
            &camera_matrix, 
            &self.distortion_coeffs,
            &Mat::eye(3, 3, opencv::core::CV_32F).unwrap(),
            &new_k,
            size,
            opencv::core::CV_16SC2,
            &mut map1,
            &mut map2
        ).unwrap();

        let mut out = image.clone();
        opencv::imgproc::remap(&image, &mut out, &map1, &map2, 
            opencv::imgproc::INTER_LINEAR,
            opencv::core::BORDER_CONSTANT,
            Default::default()
        ).unwrap();

        Ok(out)
    }

    pub fn recover_rotation(&self, frame: &FrameOpticFlow) -> Result<FramePose, CameraError> {
        // let good_points = frame.good_points();

        let mut mask = Mat::default();

        let prev_points = self.undistort_points(&frame.prev_points())?;
        let curr_points = self.undistort_points(&frame.curr_points())?;
        let essential_mat = find_essential_mat_matrix(&prev_points, &curr_points, &self.camera_matrix, RANSAC, 0.999, 0.1, &mut mask)?;

        let mut good_points = 0;
        let mut bad_points = 0;
        for m in mask.to_vec_2d::<u8>()? {
            if m.iter().copied().sum::<u8>() > 0 {
                good_points += 1;
            } else {
                bad_points += 1;
            }
        }
        if bad_points > 0 {
            trace!("points used for finding the essential matrix: {}/{}", good_points, good_points + bad_points);
        }

        let rotation = if false {
            let mut m1_mat = Mat::default();
            let mut m2_mat = Mat::default();
            let mut t = Mat::default();
            opencv::calib3d::decompose_essential_mat(&essential_mat, &mut m1_mat, &mut m2_mat, &mut t)?;
    
            let m1 = opencv_to_matrix3x3(&m1_mat);
            let m2 = opencv_to_matrix3x3(&m2_mat);
    
            let r1 = Rotation::<f64, 3>::from_matrix(&m1);
            let r2 = Rotation::<f64, 3>::from_matrix(&m2);
    
            if r1.angle() < r2.angle() {
                r1
            } else {
                r2
            }
        } else {
            let mut rotation_mat = eye();
            let mut translation_vec = Mat::zeros(3, 1, CV_32F)?.to_mat()?;
            recover_pose_camera(&essential_mat, &prev_points, &curr_points, &eye(), &mut rotation_mat, &mut translation_vec, &mut mask)?;
            let rotation_mat = opencv_to_matrix3x3(&rotation_mat);
            Rotation::<f64, 3>::from_matrix(&rotation_mat)
        };

        Ok(FramePose { points_used: good_points, rotation })
    }
}

fn eye() -> Mat {
    Mat::eye(3, 3, opencv::core::CV_32F).unwrap().to_mat().unwrap()
}
