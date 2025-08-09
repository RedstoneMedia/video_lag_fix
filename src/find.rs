use std::arch::x86_64::{__m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_sad_epu8, _mm256_setzero_si256, _mm256_storeu_si256};
use ffmpeg_sidecar::event::OutputVideoFrame;
use image::{DynamicImage, ImageFormat, RgbImage};
use image_hasher::HashAlg;
use fast_image_resize::images::Image;
use fast_image_resize::{CpuExtensions, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use std::path::{Path, PathBuf};
use ffmpeg_sidecar::command::FfmpegCommand;
use std::fs;
use std::ops::Deref;
use crate::Args;
use crate::rife::Rife;

mod lukas;
mod convolution;

#[inline]
fn get_luma(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32 * 77) + (g as u32 * 150) + (b as u32 * 29)) >> 8
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();

    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();

    let chunks = len / 32;
    let mut acc = _mm256_setzero_si256();

    for i in 0..chunks {
        // load 32 each (unaligned ok)
        let off = (i * 32) as isize;
        let va = _mm256_loadu_si256(ptr_a.offset(off) as *const __m256i);
        let vb = _mm256_loadu_si256(ptr_b.offset(off) as *const __m256i);

        // sum of absolute differences
        let sad = _mm256_sad_epu8(va, vb);
        acc = _mm256_add_epi64(acc, sad);
    }

    let mut acc_parts = [0i64; 4];
    _mm256_storeu_si256(acc_parts.as_mut_ptr() as *mut __m256i, acc);
    let mut total: u64 = acc_parts.iter().map(|&x| x as u64).sum();

    // Leftover bytes
    for i in (chunks * 32)..len {
        total += a[i].abs_diff(b[i]) as u64;
    }

    total as f32 / len as f32
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        total += a[i].abs_diff(b[i]) as u64;
    }
    total as f32 / len as f32
}


fn to_grayscale(frame: &OutputVideoFrame) -> Vec<u8> {
    let src = &frame.data;
    let n = src.len() / 3;
    let mut gray = Vec::with_capacity(n);
    unsafe { gray.set_len(n) }; // SAFETY: we immediately initialize every element
    for (i, pixel) in frame.data.chunks_exact(3).enumerate() {
        let y = get_luma(pixel[0], pixel[1], pixel[2]) as u8;
        unsafe {
            *gray.get_unchecked_mut(i) = y;
        }
    }

    gray
}

struct FrameDifference {
    hash_distance: f32,
    motion_estimate: f32
}

impl FrameDifference {
    pub const INFINITY: FrameDifference = FrameDifference {
        hash_distance: f32::INFINITY,
        motion_estimate: f32::INFINITY
    };

    pub fn is_normal(&self) -> bool {
        self.hash_distance.is_normal() && self.motion_estimate.is_normal()
    }
}

fn get_motion_estimate(a: &[u8], b: &[u8]) -> f32 {
    unsafe { mean_abs_diff(&a, &b) }
}

fn compare_frames_hash(a: &OutputVideoFrame, b: &OutputVideoFrame, args: &Args) -> FrameDifference {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    let gray_a = to_grayscale(a);
    let gray_b = to_grayscale(b);
    let motion_estimate = get_motion_estimate(&gray_a, &gray_b);

    let img_a = Image::from_vec_u8(a.width, a.height, gray_a, PixelType::U8).unwrap();
    let img_b = Image::from_vec_u8(b.width, b.height, gray_b, PixelType::U8).unwrap();

    // Setup PHash
    let hash_width = a.width / args.diff_hash_resize;
    let hash_height = a.height / args.diff_hash_resize;
    let hasher = image_hasher::HasherConfig::new()
        .hash_size(hash_width, hash_height)
        .resize_filter(image::imageops::FilterType::Nearest) // Actual resizing happened already
        .hash_alg(HashAlg::Gradient)
        .to_hasher();
    // Pre-resize because image-rs resizing in Hasher is slow.
    let mut resizer = Resizer::new();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Avx2);
    }
    let resize_options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    let width = hash_width + 1; // According to HashAlg::Gradient (row-major)
    let height = hash_height;
    let mut resized_a = DynamicImage::new(width, height, image::ColorType::L8);
    let mut resized_b = DynamicImage::new(width, height, image::ColorType::L8);
    resizer.resize(&img_a, &mut resized_a, &resize_options).unwrap();
    resizer.resize(&img_b, &mut resized_b, &resize_options).unwrap();
    // Hash
    let hash_a = hasher.hash_image(&resized_a);
    let hash_b = hasher.hash_image(&resized_b);
    let hash_distance = hash_a.dist(&hash_b) as f32 / (hash_width * hash_height) as f32;
    FrameDifference {
        hash_distance,
        motion_estimate,
    }
}

#[derive(Debug, Clone, Copy)]
struct ExponentialMovingAverage {
    pub alpha: f32,
    pub value: Option<f32>
}

impl ExponentialMovingAverage {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            value: None,
        }
    }

    pub fn commit(&mut self, value: f32) {
        self.value = match self.value {
            Some(prev) => Some(self.alpha * value + (1.0 - self.alpha) * prev),
            None => Some(value),
        };
    }
}

impl Deref for ExponentialMovingAverage {
    type Target = Option<f32>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[derive(Debug)]
pub struct DuplicateChain {
    pub frames: Vec<u32>,
    pub timestamps: Vec<f32>,
    pub frames_dir: PathBuf,
}

fn frame_to_image(frame: OutputVideoFrame) -> RgbImage {
    image::ImageBuffer::from_vec(frame.width, frame.height, frame.data).unwrap()
}

pub fn find_duplicates(args: &Args, rife: &mut Rife) {
    println!("{:?}", args);
    let iter = FfmpegCommand::new()
        .input(args.input_path.display().to_string())
        .rawvideo()
        .spawn().unwrap()
        .iter().unwrap();

    let s = std::time::Instant::now();

    let mut last_chain : Vec<OutputVideoFrame> = Vec::new();
    let mut last_diff = 0.0;
    let mut avg_hash = ExponentialMovingAverage::new(args.diff_mean_alpha);
    let mut avg_motion = ExponentialMovingAverage::new(0.7);
    let mut bg_motion = ExponentialMovingAverage::new(0.1);
    let mut chain_motion = 0.0;
    let mut i = 0;
    for frame in iter.filter_frames() {
        let diff = match last_chain.last() {
            None => FrameDifference::INFINITY,
            Some(previous) => compare_frames_hash(&previous, &frame, &args),
        };
        chain_motion += (diff.motion_estimate - bg_motion.unwrap_or(0.0)).max(0.0);

        let avg_diff = avg_hash.unwrap_or(0.0);
        let threshold = (avg_diff * args.mul_dup_threshold).min(args.max_dup_threshold);

        //println!("#{} - #{} diff: {:.4}, ema: {:.4}, t: {:.4}", frame.frame_num - 1, frame.frame_num, diff.hash_distance, avg_diff, threshold);
        if diff.hash_distance > threshold {
            if last_chain.len() > 1 {
                let clean_avg_motion = avg_motion.unwrap_or(0.0);
                let required_motion = clean_avg_motion * (last_chain.len() - 1) as f32 * 0.8;
                let c = if chain_motion >= required_motion {'X'} else {'-'};
                println!("#{} - #{} motion: {:.4}, m-ema: {:.4}, bg-ema: {:.4} m-t: {:.4}", frame.frame_num - last_chain.len() as u32, frame.frame_num, chain_motion, clean_avg_motion, bg_motion.unwrap_or(0.0), required_motion);
                if chain_motion >= required_motion {
                    println!("Achieved");
                    last_chain.clear();
                    chain_motion = 0.0;
                } else {

                }
            } else {
                last_chain.clear();
                if diff.is_normal() {
                    avg_hash.commit(diff.hash_distance);
                    avg_motion.commit(diff.motion_estimate);
                }
            }

            if last_chain.len() > 20 {
                last_chain.clear();
                chain_motion = 0.0;
                println!("Clear");
            }

            /*
            let chain = process_new_chain(&mut last_chain, &frame, args, i);
            if let Some(chain) = chain {
                println!(
                    "Duplicates {} found #{}-#{}, length: {}, diff_ema: {:0.4}, diff: {:0.4} at {:.3}s",
                    i, chain.frames[0], chain.frames.last().unwrap(), chain.frames.len(), avg_diff, last_diff, chain.timestamps[0]
                );

                i += 1;
                let patch_dir = format!("tmp/patch_{}", i);
                fs::create_dir_all(&patch_dir).expect("Failed to create patch dir");
                rife.generate_in_betweens(
                    chain,
                    &patch_dir,
                );
            }

            if diff.is_normal() {
                avg_hash.commit(diff.hash_distance);
                avg_motion.commit(diff.motion_estimate);
            }
             */
        }
        /*
        let chain_length = last_chain.len();
        if chain_length > args.max_duplicates {
            // Free up memory from too long chains using dummy frames
            for frame in last_chain.iter_mut().take(chain_length) {
                frame.data = Vec::with_capacity(0);
            }
        }*/

        if diff.is_normal() {
            bg_motion.commit(diff.motion_estimate);
        }

        last_chain.push(frame);
        last_diff = diff.hash_distance;
    }

    println!("Finding Duplicates Took: {}s", s.elapsed().as_secs_f32());
}

fn process_new_chain(last_chain: &mut Vec<OutputVideoFrame>, end_frame: &OutputVideoFrame, args: &Args, i: i32) -> Option<DuplicateChain> {
    if args.min_duplicates > last_chain.len() || last_chain.len() > args.max_duplicates {
        last_chain.clear();
        return None;
    }
    let frames: Vec<_> = last_chain.iter().map(|f| f.frame_num).collect();
    let timestamps: Vec<_> = last_chain.iter().map(|f| f.timestamp).collect();

    let start_image = frame_to_image(last_chain.remove(0));
    let end_image = frame_to_image(end_frame.clone());

    let frames_path = Path::new("tmp/frames");
    let dir = frames_path.join(i.to_string());
    fs::create_dir_all(&dir).unwrap();

    let start_frame_path = dir.join("0.webp").to_string_lossy().to_string();
    let end_frame_path = dir.join("1.webp").to_string_lossy().to_string();
    start_image.save_with_format(&start_frame_path, ImageFormat::WebP).unwrap();
    end_image.save_with_format(&end_frame_path, ImageFormat::WebP).unwrap();
    last_chain.clear();

    let chain = DuplicateChain {
        frames,
        timestamps,
        frames_dir: dir
    };
    Some(chain)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_abs_diff_small() {
        let a = vec![0u8, 50, 100, 150, 200, 250];
        let b = vec![5u8, 45, 110, 140, 190, 253];
        let expected: f32 = (5 + 5 + 10 + 10 + 10 + 3) as f32 / 6.0;
        unsafe {
            let got = mean_abs_diff(&a, &b);
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mean_abs_diff_random() {
        let n = 3_686_413;
        let a: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let b: Vec<u8> = (0..n).rev().map(|i| (i % 256) as u8).collect();
        let scalar: f32 = a.iter().zip(b.iter())
            .map(|(&x,&y)| (x as i16 - y as i16).abs() as u32)
            .sum::<u32>() as f32
            / n as f32;
        unsafe {
            let simd = mean_abs_diff(&a, &b);
            assert!((simd - scalar).abs() < 1e-6);
        }
    }
}