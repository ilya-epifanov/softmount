use nalgebra::{Matrix3, Vector3};
use opencv::prelude::{Mat, MatTrait};
use rand::prelude::IteratorRandom;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

pub fn opencv_to_matrix3x3(mat: &Mat) -> Matrix3<f64> {
    assert_eq!(mat.rows(), 3);
    assert_eq!(mat.cols(), 3);

    Matrix3::new(
        *mat.at_2d(0, 0).unwrap(), *mat.at_2d(0, 1).unwrap(), *mat.at_2d(0, 2).unwrap(), 
        *mat.at_2d(1, 0).unwrap(), *mat.at_2d(1, 1).unwrap(), *mat.at_2d(1, 2).unwrap(), 
        *mat.at_2d(2, 0).unwrap(), *mat.at_2d(2, 1).unwrap(), *mat.at_2d(2, 2).unwrap(),
    )
}

#[allow(unused)]
pub fn opencv_to_vector3(mat: &Mat) -> Vector3<f64> {
    assert_eq!(mat.rows(), 3);
    assert_eq!(mat.cols(), 1);

    Vector3::new(
        *mat.at_2d(0, 0).unwrap(), *mat.at_2d(1, 0).unwrap(), *mat.at_2d(2, 0).unwrap(), 
    )
}

#[allow(unused)]
pub fn dump_csv(output_path: &str, data: &[(&str, &[f64])]) {
    let mut out = BufWriter::new(File::create(output_path).unwrap());
    for (ix, (c, _)) in data.iter().copied().enumerate() {
        if ix != 0 {
            let _ = write!(out, ",");
        }
        let _ = write!(out, "{}", c);
    }
    let _ = writeln!(out);

    let length = data[0].1.len();
    for i in 0..length {
        for (j, (_, d)) in data.iter().copied().enumerate() {
            if j != 0 {
                let _ = write!(out, ",");
            }
            let _ = write!(out, "{}", d[i]);
        }
        let _ = writeln!(out);
    }
}

#[allow(unused)]
pub fn sliding_average(v: &[f64], window_size: usize) -> Vec<f64> {
    if window_size <= 1 {
        return v.iter().copied().collect();
    }

    let scale = 1.0 / window_size as f64;
    let mut window = vec![0.0; window_size];
    let mut ret = Vec::with_capacity(v.len() - window_size + 1);
    for ix in 0..window_size {
        window.push(v[ix]);
    }
    ret.push(window.iter().copied().sum::<f64>() * scale);

    for ix in window_size..v.len() {
        window[ix % window_size] = v[ix];
        ret.push(window.iter().copied().sum::<f64>() * scale);
    }

    assert_eq!(ret.len(), v.len() - window_size + 1);
    ret
}

pub fn ransac<T: Default + Copy, ScoreModel, const N: usize>(vs: &[T], inlier_prob: f64, confidence: f64, score_model: ScoreModel) -> Vec<bool> where ScoreModel: Fn([T; N], &[T], &mut [bool]) -> f64 {
    let good_sample_prob = inlier_prob.powi(N as i32);
    let attempts = ((1.0 - confidence).ln() / (1.0 - good_sample_prob).ln() + 3.0 * (1.0 - good_sample_prob).sqrt() / good_sample_prob).ceil() as usize;

    let mut rng = rand::thread_rng();

    let mut mask = vec![false; vs.len()];
    let mut best_inliers = vec![false; vs.len()];
    let mut best_score = f64::MIN;

    for _ in 0..attempts {
        let mut values = [T::default(); N];

        for (v_ix, (ix, _v)) in vs.iter().enumerate().choose_multiple(&mut rng, N).into_iter().enumerate() {
            values[v_ix] = vs[ix];
        }

        mask.fill(false);
        let score = score_model(values, vs, &mut mask);
        if score > best_score {
            best_score = score;
            std::mem::swap(&mut mask, &mut best_inliers);
        }
    }

    best_inliers
}

