use crate::env::{cell, Hex, A, B, C, D, E};

mod env;

fn main() {
    println!("Hello, hex!");
    const N: usize = 5;
    let mut hex = Hex::<N>::default();
    for cell_take in [
        cell(E, 1), cell(A, 1),
        cell(D, 2), cell(A, 1),
        cell(C, 3), cell(A, 1),
        cell(D, 3), cell(A, 1),
        cell(E, 3), cell(A, 1),
        cell(E, 4), cell(A, 1),
        cell(C, 2), cell(A, 1),
        cell(B, 2), cell(A, 1),
        cell(A, 2), cell(A, 1),
        cell(A, 3), cell(A, 1),
        cell(A, 4), cell(A, 1),
        cell(B, 4), cell(A, 1),
        cell(D, 5),
    ].into_iter() {
        let next = hex.next();
        _ = hex.next_take_cell(cell_take);
        println!("{next} {cell_take}\n{}\n", hex);
    }
    hex.undo()
}
