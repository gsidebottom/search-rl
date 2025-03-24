use std::default::Default;
use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::iter::once;
use itertools::Itertools;
use Player::{Blue, Red};
use search_rl::env::{Action, State};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    Red,
    Blue,
}

impl Display for Player {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Red => "R",
            Blue => "B",
        })
    }
}

impl Player {
    pub fn other(&self) -> Player {
        match self { Red => Blue, Blue => Red }
    }

    pub fn occupies<const N: usize>(&self, board: &Board<N>, cell: &Cell) -> bool {
        board
            .at(cell)
            .as_ref()
            .map(|player_at_cell| player_at_cell == self)
            .unwrap_or_default()
    }

    fn start_cell_iter<'a, const N: usize>(&'a self, board: &'a Board<N>) -> impl Iterator<Item = Cell> + 'a {
        (0..N)
            .filter_map(move |k| {
                let cell = match *self {
                    Red => Cell(k, 0),
                    Blue => Cell(0, k),
                };
                self.occupies(board, &cell).then_some(cell)
            })
    }

    fn is_end<const N: usize>(&self, Cell(i, j): &Cell) -> bool {
        match self {
            Red => *j == N - 1,
            Blue => *i == N - 1
        }
    }

    fn connect_iter<'a, const N: usize>(
        &'a self,
        board: &'a Board<N>,
        cell: &'a Cell,
    ) -> impl Iterator<Item = Cell> + 'a {
        board.adj_iter(cell).filter(|cell| self.occupies(board, cell))
    }

    fn wins<'a, const N: usize>(
        &'a self,
        board: &'a Board<N>,
    ) -> bool {
        let mut seen = HashSet::new();
        self.start_cell_iter(board)
            .any(|start_cell| {
                self.wins_from(board, &start_cell, &mut seen)
            })
    }

    fn wins_from<'a, const N: usize>(
        &'a self,
        board: &'a Board<N>,
        cell: &'a Cell,
        seen:  &mut HashSet<Cell>,
    ) -> bool {
        if seen.contains(cell) {
            return false
        }
        if !self.occupies(board, cell) {
            return false
        }
        if self.is_end::<N>(cell) {
            return true
        }
        seen.insert(*cell);
        if self.connect_iter(board, cell)
            .any(|cell| self.wins_from(board, &cell, seen)) {
            return true
        }
        seen.remove(cell);
        false
    }
}

pub fn i_char(i: usize) -> char {
    (b'A' + i as u8) as char
}

pub fn j_char(j: usize) -> char {
    (b'1' + j as u8) as char
}

pub const A: usize = 0;
pub const B: usize = 1;
pub const C: usize = 2;
pub const D: usize = 3;
pub const E: usize = 4;
pub const _F: usize = 5;
pub const _G: usize = 6;
pub const _H: usize = 7;
pub const _I: usize = 8;
pub const _J: usize = 9;
pub const _K: usize = 10;
pub const _L: usize = 11;
pub const _M: usize = 12;
pub const _N: usize = 13;
pub const _O: usize = 14;
pub const _P: usize = 15;
pub const _Q: usize = 16;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cell(pub usize, pub usize);

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let Self(i,j) = self;
        write!(f, "{}{}", i_char(*i), j_char(*j))
    }
}

impl Cell {
    pub fn new((i, j): (usize, usize)) -> Self {
        Self(i, j)
    }
}

pub fn cell(i: usize, j_plus_one: usize) -> Cell { Cell::new((i, j_plus_one-1)) }
#[derive(Debug, Clone)]
pub struct Board<const N: usize>([[Option<Player>; N]; N]);

impl <const N: usize> Display for Board<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let board_string =
            once(
                once("   ".to_string())
                    .chain(
                        (0..N).map(|i| format!("{:2}", i_char(i)))
                    ).collect::<Vec<_>>().join("")
            )
                .chain(
                    (0..N).map(|j| {
                        once(format!("{}{:2}", " ".repeat(j), j_char(j)))
                            .chain(
                                (0..N)
                                    .map(|i| {
                                        self.cell_to_string(&Cell(i, j))
                                    })
                            )
                            .collect::<Vec<_>>()
                            .join(" ")
                    }))
                .collect::<Vec<_>>()
                .join("\n");
        write!(f, "{board_string}")
    }
}
impl<const N: usize> Default for Board<N> {
    fn default() -> Self {
        Self([[None; N]; N])
    }
}

impl<const N: usize> Board<N> {
    pub fn at(&self, Cell(i, j): &Cell) -> &Option<Player> {
        &self.0[*i][*j]
    }

    pub fn at_mut(&mut self, Cell(i, j): &Cell) -> &mut Option<Player> {
        &mut self.0[*i][*j]
    }

    pub fn set(&mut self, cell: &Cell, player: Player) -> Option<Player> {
        self.at_mut(cell).replace(player)
    }

    pub fn clear(&mut self, cell: &Cell) {
        *self.at_mut(cell) = None
    }

    pub fn cell_to_string(&self, cell: &Cell) -> String {
        self
            .at(cell)
            .map(|player| player.to_string())
            .unwrap_or("_".to_string())
    }

    fn adj_iter(&self, Cell(i, j): &Cell) -> impl Iterator<Item = Cell> + '_ {
        let (i, j) = (*i, *j);
        let up_right = (i < N - 1 && j > 0)
            .then_some(())
            .map(|_| Cell(i + 1, j - 1));
        let up = (j > 0)
            .then_some(())
            .map(|_| Cell(i, j - 1));
        let left = (i > 0)
            .then_some(())
            .map(|_| Cell(i - 1, j));
        let right = (i < N - 1)
            .then_some(())
            .map(|_| Cell(i + 1, j));
        let down_left = (i > 0 && j < N - 1)
            .then_some(())
            .map(|_| Cell(i - 1, j + 1));
        let down = (j < N - 1)
            .then_some(())
            .map(|_| Cell(i, j + 1));
        [up_right, up, left, right, down_left, down]
            .into_iter()
            .flat_map(|next| next.into_iter())
    }

    pub fn as_array(&self) -> [[i32; N]; N] {
        self.0
            .iter()
            .map(|col| col
                .iter()
                .map(|c| match c {
                    None => 0,
                    Some(Red) => 1,
                    Some(Blue) => -1,
                })
                .collect_array().unwrap()
            )
            .collect_array().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct Hex<const N: usize> {
    board: Board<N>,
    empty_cells: HashSet<Cell>,
    taken: Vec<Cell>,
    next: Player,
    winner: Option<Player>,
}

impl<const N: usize> Default for Hex<N> {
    fn default() -> Self {
        Self {
            board: Default::default(),
            empty_cells: (0..N).zip(0..N).map(Cell::new).collect(),
            taken: Vec::new(),
            next: Red,
            winner: None,
        }
    }
}

impl<const N: usize> Hex<N> {

    pub fn board(&self) -> &Board<N> {
        &self.board
    }

    pub fn next(&self) -> Player {
        self.next
    }

    #[allow(dead_code)]
    pub fn empty_cell_iter(&self) -> impl Iterator<Item = Cell> + '_ {
        self.empty_cells.iter().copied()
    }

    pub fn next_take_cell(&mut self, cell: Cell) -> bool {
        _ = self.empty_cells.remove(&cell);
        _ = self.board.set(&cell, self.next);
        self.taken.push(cell);
        self.winner = self.next.wins(&self.board).then_some(self.next);
        self.next = self.next.other();
        self.winner.is_some()
    }

    /// "Draws are impossible in Hex due to the topology of the game board", but anyway...
    pub fn is_draw(&self) -> bool {
        self.winner.is_none() && self.empty_cells.is_empty()
    }

    pub fn winner(&self) -> Option<Player> {
        self.winner
    }

    pub fn undo(&mut self) {
        while let Some(cell) = self.taken.pop() {
            self.board.clear(&cell);
            _ = self.empty_cells.insert(cell);
            self.next = self.next.other()
        }
    }
}

impl<const N: usize> Display for Hex<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let result_str = match (self.winner(), self.is_draw()) {
            (Some(player), _) => &format!("{player}, wins"),
            (_, true) => "draw",
            _ => "mid game"
        };
        write!(f, "{}\n{result_str}", self.board())
    }
}

impl<const N: usize> State<N> for Hex<N> {
    fn init() -> Self {
        Self::default()
    }

    fn action_count(&self) -> usize {
        self.taken.len()
    }

    fn take(&self, action: Action) -> Self {
        let cell = self.taken[action.index()];
        let mut moved = self.clone();
        moved.next_take_cell(cell);
        moved
    }

    fn reward(&self) -> Option<f32> {
        if self.is_draw() {
            return Some(0.1)
        }
        if let Some(player) = self.winner() {
            return Some(match player { Red => 1.0, Blue => -1.0 })
        }
        None
    }

    fn value(&self, _taken: Action, value: f32) -> f32 {
        if value != 0.1 { -value } else { value }
    }

    fn as_array(&self) -> [[i32; N]; N] {
        self.board.as_array()
    }
}