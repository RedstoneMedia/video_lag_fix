use crate::find::mean_abs_diff;

pub struct EstimatedMotion {
    vx: f32,
    vy: f32,
    eigen1: f32,
    eigen2: f32,
}

const R: isize     = 32;   // search radius
const INF_SCORE: f32 = f32::MAX;

const LDSP: [(isize, isize); 8] = [
    ( 0, -2), (-1, -1), (1, -1),
    (-2,  0),  ( 2,  0),
    (-1,  1), (1,  1), ( 0,  2),
];

const SDSP: [(isize, isize); 5] = [
    ( 0, -1), (-1,  0),
    (0, 0),
    ( 1,  0), ( 0,  1),
];


pub struct PatchEstimationData<'a> {
    img1: &'a [u8],
    img2: &'a [u8],
    width: usize,
    height: usize,
    x_center: usize,
    y_center: usize,
    initial_dx: isize,
    initial_dy: isize,
}

pub fn estimate_patch_motion<const PS: usize, const SR: usize>(
    data: PatchEstimationData
) -> Option<EstimatedMotion> {
    let half_patch = PS / 2;

    // Ensure search area fits inside the image
    if data.x_center < SR + half_patch
        || data.y_center < SR + half_patch
        || data.x_center + SR + half_patch >= data.width
        || data.y_center + SR + half_patch >= data.height
    {
        return None;
    }

    let mut best_dx = data.initial_dx;
    let mut best_dy = data.initial_dy;
    let mut best_score = compute_ssd::<PS>(&data, best_dx, best_dy);

    // Large Diamond Search
    let mut changed = true;
    while changed {
        changed = search_once::<PS, SR>(&data, &mut best_dx, &mut best_dy, &mut best_score, &LDSP);
        if changed {
            //println!("New best: {}", best_score);
        }
    }
    // Small Diamond Refinement step
    changed = search_once::<PS, SR>(&data, &mut best_dx, &mut best_dy, &mut best_score, &SDSP);
    if changed {
        //println!("New optim best: {}", best_score);
    }

    Some(EstimatedMotion {
        vx: best_dx as f32,
        vy: best_dy as f32,
        eigen1: 1.0,
        eigen2: 1.0,
    })
}

fn search_once<const PS: usize, const SR: usize>(
    data: &PatchEstimationData,
    best_dx: &mut isize,
    best_dy: &mut isize,
    best_score: &mut f32,
    pattern: &[(isize, isize)],
) -> bool {
    let mut changed = false;

    let mut current_best_dx = *best_dx;
    let mut current_best_dy = *best_dy;

    for (dx, dy) in pattern {
        let new_dx = *best_dx + dx;
        let new_dy = *best_dy + dy;

        if new_dx.abs() > SR as isize || new_dy.abs() > SR as isize {
            continue;
        }

        let score = compute_ssd::<PS>(
            data,
            new_dx,
            new_dy,
        );

        if score < *best_score {
            *best_score = score;
            current_best_dx = new_dx;
            current_best_dy = new_dy;
            changed = true;
            //println!("New best: {}", score);
            //break; // Greedy: jump to better neighbor
        }
    }
    *best_dx = current_best_dx;
    *best_dy = current_best_dy;

    changed
}

fn compute_ssd<const PS: usize>(
    data: &PatchEstimationData,
    dx: isize,
    dy: isize,
) -> f32 {
    let half_patch = PS / 2;
    let start_x1 = data.x_center - half_patch;
    let start_y1 = data.y_center - half_patch;

    let start_x2 = start_x1.wrapping_add_signed(dx);
    let start_y2 = start_y1.wrapping_add_signed(dy);

    let start_i1 = start_y1 * data.width + start_x1;
    let start_i2 = start_y2 * data.width + start_x2;
    let end_i1 = start_i1 + PS;
    let end_i2 = start_i2 + PS;

    let s1 = &data.img1[start_i1..=end_i1];
    let s2 = &data.img2[start_i2..=end_i2];
    unsafe {mean_abs_diff(&s1, &s2)}
}

fn estimate_total_motion<const PS: usize, const SR: usize>(
    img1: &[u8],
    img2: &[u8],
    width: usize,
    height: usize,
    grid: (usize, usize),
    eigen2_threshold: f32,
) -> Option<f32> {
    let mut total_weight = 0.0f32;
    let mut weighted_sum = 0.0f32;

    let step_x = width / (grid.0 + 1);
    let step_y = height / (grid.1 + 1);

    for gy in 1..=grid.1 {
        for gx in 1..=grid.0 {
            let cx = gx * step_x;
            let cy = gy * step_y;

            let Some(est) = estimate_patch_motion::<PS, SR>(
                PatchEstimationData {
                    img1,
                    img2,
                    width,
                    height,
                    x_center: cx,
                    y_center: cy,
                    initial_dx: 0,
                    initial_dy: 0,
                }
            ) else {continue};

            if est.eigen2 > eigen2_threshold {
                let mag = (est.vx * est.vx + est.vy * est.vy).sqrt();
                let weight = (est.eigen2 - eigen2_threshold).max(0.0);
                weighted_sum += weight * mag;
                total_weight += weight;
            }
        }
    }

    if total_weight > 0.0 {
        Some(weighted_sum / total_weight)
    } else {
        None
    }
}


#[cfg(test)]
mod tests {
    use std::thread::scope;
    use super::*;
    use image::{RgbaImage, Rgba, GenericImageView, DynamicImage};
    use imageproc::drawing::{draw_antialiased_polygon_mut, draw_hollow_circle_mut, draw_hollow_rect_mut};
    use imageproc::pixelops::interpolate;
    use imageproc::point::Point;
    use imageproc::rect::Rect;

    fn draw_thick_arrow(
        img: &mut RgbaImage,
        (x0, y0): (f32, f32),
        (x1, y1): (f32, f32),
        color: Rgba<u8>,
        thickness: f32,
        head_len: f32,
        head_width: f32,
    ) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();

        if len < 1e-3 {
            return; // too short to draw
        }

        let ux = dx / len;
        let uy = dy / len;
        let px = -uy;
        let py = ux;
        let half = thickness / 2.0;

        // Shaft rectangle points
        let sx0 = x0 + px * half;
        let sy0 = y0 + py * half;
        let sx1 = x0 - px * half;
        let sy1 = y0 - py * half;
        let ex0 = x1 - ux * head_len + px * half;
        let ey0 = y1 - uy * head_len + py * half;
        let ex1 = x1 - ux * head_len - px * half;
        let ey1 = y1 - uy * head_len - py * half;

        let shaft = vec![
            Point::new(sx0.round() as i32, sy0.round() as i32),
            Point::new(ex0.round() as i32, ey0.round() as i32),
            Point::new(ex1.round() as i32, ey1.round() as i32),
            Point::new(sx1.round() as i32, sy1.round() as i32),
        ];

        // Only draw shaft if not degenerate
        if shaft.windows(2).all(|w| w[0] != w[1]) {
            draw_antialiased_polygon_mut(img, &shaft, color, interpolate);
        }

        // Arrowhead triangle
        let base0 = (
            x1 - ux * head_len + px * head_width / 2.0,
            y1 - uy * head_len + py * head_width / 2.0,
        );
        let base1 = (
            x1 - ux * head_len - px * head_width / 2.0,
            y1 - uy * head_len - py * head_width / 2.0,
        );

        let head = vec![
            Point::new(x1.round() as i32, y1.round() as i32),
            Point::new(base0.0.round() as i32, base0.1.round() as i32),
            Point::new(base1.0.round() as i32, base1.1.round() as i32),
        ];

        if head.windows(2).all(|w| w[0] != w[1]) {
            draw_antialiased_polygon_mut(img, &head, color, interpolate);
        }
    }

    const PATCH_SIZE: usize = 24;
    const SEARCH_RADIUS: usize = 80;

    #[test]
    fn test_draw_motion_grid() {
        // Load images
        let img1 = image::open("tests/frames/0006.png").unwrap();
        let img2 = image::open("tests/frames/0007.png").unwrap();
        let (width, height) = img1.dimensions();
        assert_eq!(img2.dimensions(), (width, height));

        let gray1 = img1.to_luma8();
        let gray2 = img2.to_luma8();

        // Create an RGBA image for overlay
        let mut overlay = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));

        let color = Rgba([255, 0, 0, 128]);
        let rect_color = Rgba([0, 255, 0, 120]);

        let half_patch = PATCH_SIZE / 2;

        for y in (0..height as usize).step_by(PATCH_SIZE) {
            for x in (0..width as usize).step_by(PATCH_SIZE) {
                let rect = Rect::at(x as i32, y as i32).of_size(PATCH_SIZE as u32, PATCH_SIZE as u32);
                draw_hollow_rect_mut(&mut overlay, rect, rect_color);

                let x = x + half_patch;
                let y = y + half_patch;

                if let Some(motion) = estimate_patch_motion::<PATCH_SIZE, SEARCH_RADIUS>(
                    PatchEstimationData {
                        img1: &gray1,
                        img2: &gray2,
                        width: width as usize,
                        height: height as usize,
                        x_center: x,
                        y_center: y,
                        initial_dx: 0,
                        initial_dy: 0,
                    }
                ) {
                    println!("{:.04} {:.04}", motion.vx, motion.vy);
                    let x0 = x as f32;
                    let y0 = y as f32;
                    let x1 = x0 + motion.vx;
                    let y1 = y0 + motion.vy;
                    draw_thick_arrow(&mut overlay, (x0, y0), (x1, y1), color, 2.0, 7.0, 7.0);
                }
            }
        }
        let mut base = DynamicImage::from(gray1).to_rgba8();
        let mut new = DynamicImage::from(gray2).to_rgba8();
        for n in new.pixels_mut() {
            n.0[3] = 100;
        }
        image::imageops::overlay(&mut base, &new, 0, 0);
        image::imageops::overlay(&mut base, &overlay, 0, 0);

        base.save("tests/out_motion_debug.png").unwrap();
    }


    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r1, g1, b1) = match h as u32 {
            0..=59 => (c, x, 0.0),
            60..=119 => (x, c, 0.0),
            120..=179 => (0.0, c, x),
            180..=239 => (0.0, x, c),
            240..=299 => (x, 0.0, c),
            300..=359 => (c, 0.0, x),
            _ => (0.0, 0.0, 0.0),
        };

        let r = ((r1 + m) * 255.0).round() as u8;
        let g = ((g1 + m) * 255.0).round() as u8;
        let b = ((b1 + m) * 255.0).round() as u8;

        (r, g, b)
    }

    #[test]
    fn visual_one_patch() {
        let img1 = image::open("tests/frames/0006.png").unwrap();
        let img2 = image::open("tests/frames/0007.png").unwrap();
        let (width, height) = img1.dimensions();
        assert_eq!(img2.dimensions(), (width, height));

        let gray1 = img1.to_luma8();
        let gray2 = img2.to_luma8();

        let s = 2;
        let width = width / s;
        let height = height / s;
        let gray1 = image::imageops::resize(&gray1, width, height, image::imageops::FilterType::Lanczos3);
        let gray2 = image::imageops::resize(&gray2, width, height, image::imageops::FilterType::Lanczos3);

        // Create an RGBA image for overlay
        let mut overlay = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));

        let rect_color = Rgba([0, 255, 0, 120]);
        let circle_color = Rgba([0, 255, 255, 120]);
        let look_color = Rgba([255, 255, 0, 120]);

        let half_patch = PATCH_SIZE / 2;

        for y in (80..(height - 80) as usize).step_by(PATCH_SIZE * 4) {
            for x in (80..(width - 80) as usize).step_by(PATCH_SIZE * 4) {
                let cx = x + half_patch;
                let cy = y + half_patch;

                let data = PatchEstimationData {
                    img1: &gray1,
                    img2: &gray2,
                    width: width as usize,
                    height: height as usize,
                    x_center: cx as usize,
                    y_center: cy as usize,
                    initial_dx: 0,
                    initial_dy: 0,
                };
                println!("{} {}", cx, cy);

                let rect = Rect::at(cx as i32 - half_patch as i32, cy as i32 - half_patch as i32).of_size(PATCH_SIZE as u32, PATCH_SIZE as u32);
                draw_hollow_rect_mut(&mut overlay, rect, rect_color);
                draw_hollow_circle_mut(&mut overlay, (cx as i32, cy as i32), SEARCH_RADIUS as i32 / s as i32, circle_color);

                let steps = 4;
                let step_distance = (SEARCH_RADIUS as f32 / s as f32 / steps as f32).ceil() as usize;
                let mut coarse_pattern = Vec::new();
                for x in -(steps as isize)..(steps as isize) {
                    for y in -(steps as isize)..(steps as isize) {
                        let dx = x * step_distance as isize;
                        let dy = y * step_distance as isize;
                        coarse_pattern.push((dx, dy));
                    }
                }

                /*
                for (dx, dy) in &coarse_pattern {
                    let x = cx as isize + dx;
                    let y = cy as isize + dy;
                    overlay.put_pixel(x as u32, y as u32, look_color);
                }*/

                let mut best_dx = 0;
                let mut best_dy = 0;
                let mut best_score = compute_ssd::<PATCH_SIZE>(&data, 0, 0);
                println!("{}", best_score);

                let mut pattern: &[(isize, isize)] = coarse_pattern.as_ref();

                let mut iters = 0;
                let mut changed = true;
                while changed {
                    let last_dx = best_dx;
                    let last_dy = best_dy;
                    changed = search_once::<PATCH_SIZE, SEARCH_RADIUS>(&data, &mut best_dx, &mut best_dy, &mut best_score, pattern);
                    if changed {
                        pattern = &LDSP;
                        let start = (last_dx as i32 + cx as i32, last_dy as i32 + cy as i32);
                        let end = (best_dx as i32 + cx as i32, best_dy as i32 + cy as i32);

                        let hue = (iters as f32 * (360.0 / 10.0)) % 360.0;
                        let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);

                        let color = Rgba([r, g, b, 128]);
                        println!("{:?} {:?} {}", start, end, best_score);
                        overlay = imageproc::drawing::draw_antialiased_line_segment(&overlay, start, end, color, interpolate);
                        iters += 1;
                    }
                }
            }
        }
        /*
        let cx = 1339 / s;
        let cy = 589 / s;
        let data = PatchEstimationData {
            img1: &gray1,
            img2: &gray2,
            width: width as usize,
            height: height as usize,
            x_center: cx as usize,
            y_center: cy as usize,
            initial_dx: 0,
            initial_dy: 0,
        };

        let rect = Rect::at(cx as i32 - half_patch as i32, cy as i32 - half_patch as i32).of_size(PATCH_SIZE as u32, PATCH_SIZE as u32);
        draw_hollow_rect_mut(&mut overlay, rect, rect_color);
        draw_hollow_circle_mut(&mut overlay, (cx as i32, cy as i32), SEARCH_RADIUS as i32 / s as i32, circle_color);

        let steps = 3;
        let step_distance = (SEARCH_RADIUS as f32 / s as f32 / steps as f32).ceil() as usize;
        let mut coarse_pattern = Vec::new();
        for x in -(steps as isize)..(steps as isize) {
            for y in -(steps as isize)..(steps as isize) {
                let dx = x * step_distance as isize;
                let dy = y * step_distance as isize;
                coarse_pattern.push((dx, dy));
            }
        }

        for (dx, dy) in &coarse_pattern {
            let x = cx as isize + dx;
            let y = cy as isize + dy;
            overlay.put_pixel(x as u32, y as u32, look_color);
        }

        let mut best_dx = 0;
        let mut best_dy = 0;
        let mut best_score = compute_ssd::<PATCH_SIZE>(&data, 0, 0);

        let mut pattern: &[(isize, isize)] = coarse_pattern.as_ref();

        let mut iters = 0;
        let mut changed = true;
        while changed {
            let last_dx = best_dx;
            let last_dy = best_dy;
            changed = search_once::<PATCH_SIZE, SEARCH_RADIUS>(&data, &mut best_dx, &mut best_dy, &mut best_score, pattern);
            if changed {
                pattern = &LDSP;
                let start = (last_dx as i32 + cx as i32, last_dy as i32 + cy as i32);
                let end = (best_dx as i32 + cx as i32, best_dy as i32 + cy as i32);

                let hue = (iters as f32 * (360.0 / 10.0)) % 360.0;
                let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);

                let color = Rgba([r, g, b, 128]);
                println!("{:?} {:?} {}", start, end, best_score);
                overlay = imageproc::drawing::draw_antialiased_line_segment(&overlay, start, end, color, interpolate);
                iters += 1;
            }
        }*/

        let mut base = DynamicImage::from(gray1).to_rgba8();

        let mut new = DynamicImage::from(gray2).to_rgba8();
        for n in new.pixels_mut() {
            n.0[3] = 100;
        }
        image::imageops::overlay(&mut base, &new, 0, 0);
        image::imageops::overlay(&mut base, &overlay, 0, 0);

        base.save("tests/out_track_one.png").unwrap();
    }

}