// Count the pinned generator's line-count RNG stream without reading the dataset.
use tpchgen::generators::OrderGenerator;

fn main() {
    let scale: u32 = std::env::args().nth(1).expect("scale factor").parse().unwrap();
    assert!(scale > 0);
    let orders = OrderGenerator::calculate_row_count(f64::from(scale), 1, 1);
    let mut random = OrderGenerator::create_line_count_random();
    let mut rows = 0_u64;
    for _ in 0..orders {
        rows += random.next_value() as u64;
        random.row_finished();
    }
    println!("{rows}");
}
