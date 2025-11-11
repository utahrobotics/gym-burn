use std::collections::hash_map::Entry;

use dbscan::Classification;
use kneed::knee_locator::{InterpMethod, KneeLocatorParams, ValidCurve, ValidDirection};
use rayon::{iter::{IntoParallelRefIterator, ParallelIterator}, slice::ParallelSliceMut};
use rustc_hash::FxHashMap;

pub struct ClusteringResult {
    pub centers: Vec<(
        [f64; 3],
        usize
    )>,
    pub unknowns: usize
}

pub fn cluster(points: Vec<[f64; 3]>, min_points: usize) -> ClusteringResult {
    let mut distances: Vec<_> = points
        .par_iter()
        .copied()
        .map(|a| {
            let mut distances: Vec<_> = points.par_iter()
                .copied()
                .map(|b| {
                    (a[0] - b[0]).powi(2) +
                    (a[1] - b[1]).powi(2) +
                    (a[2] - b[2]).powi(2)
                })
                .collect();
            distances.par_sort_unstable_by(|a, b| a.total_cmp(b));
            //  get the kth closest neighbor
            distances[min_points]
        })
        .collect();
    
    distances.par_sort_unstable_by(|a, b| a.total_cmp(b));

    let kl = kneed::knee_locator::KneeLocator::new(
        (0..distances.len()).into_iter().map(|n| n as f64).collect(),
        distances.clone(),
        1.0,
        KneeLocatorParams::new(
            ValidCurve::Convex,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        )
    ).unwrap();

    let eps = distances[kl.elbow().unwrap() as usize];

    let classifications = dbscan::cluster(eps, min_points, &points.iter().map(|p| p.to_vec()).collect());

    let mut result = ClusteringResult {
        centers: vec![],
        unknowns: 0,
    };

    let mut classified: FxHashMap<usize, ([f64; 3], usize)> = FxHashMap::default();

    for (classification, point) in classifications.into_iter().zip(points) {
        match classification {
            Classification::Core(id) | Classification::Edge(id) => {
                match classified.entry(id) {
                    Entry::Occupied(mut entry) => {
                        let (sum, count) = entry.get_mut();
                        sum[0] += point[0];
                        sum[1] += point[1];
                        sum[2] += point[2];
                        *count += 1;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert((point, 1));
                    }
                }
            }
            Classification::Noise => {
                result.unknowns += 1;
            }
        }
    }

    result.centers = classified.into_iter().map(|(_, (mut sum, count))| {
        sum[0] /= count as f64;
        sum[1] /= count as f64;
        sum[2] /= count as f64;
        (sum, count)
    }).collect();

    result
}