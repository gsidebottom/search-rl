use std::default::Default;

use strum_macros::EnumIter;
use crate::env::Action::{Claim0, Claim1, Claim2, Color01, Color12};
use crate::env::Player::{Blue, Red};

#[derive(Debug, EnumIter, Default, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
enum Player {
    #[default]
    Red,
    Blue,
}

impl Player {
    fn other(&self) -> Self {
        match *self {
            Red => Blue,
            Blue => Red,
        }
    }

    fn has_claim<const N: usize>(
        &self,
        nodes: [&Option<Player>; N],
    ) -> bool {
        nodes
            .iter()
            .any(|node|
                node.map(|player| &player != self)
                    .unwrap_or_default())
    }
}

#[derive(Debug, EnumIter, Clone, Copy)]
enum Color {
    Grey,
    Green,
    Purple,
}

#[derive(Debug, Clone, Copy)]
enum Action {
    Claim0(Player),
    Claim1(Player),
    Claim2(Player),
    Color01(Color),
    Color12(Color),
}

#[derive(Debug, Default, Clone, Copy)]
struct Network {
    node0: Option<Player>,
    node1: Option<Player>,
    node2: Option<Player>,
    link01: Option<Color>,
    link12: Option<Color>,
}

impl Network {

    fn take_action(&self, action: Action) -> Option<Self> {
        match (self, action) {
            (
                Self { node0: None, ..},
                Claim0(player)
            )  if !player.has_claim([&self.node1, &self.node2]) => {
                Some(Self {
                        node0: Some(player),
                        ..*self
                })
            }
            (
                Self { node1: None, ..},
                Claim1(player)
            ) if !player.has_claim([&self.node0, &self.node2]) => {
                Some(Self {
                    node1: Some(player),
                    ..*self
                })
            }
            (
                Self { node2: None, ..},
                Claim2(player)
            ) if !player.has_claim([&self.node0, &self.node1]) => {
                Some(Self {
                    node2: Some(player),
                    ..*self
                })
            }
            (Self { link01: None, ..}, Color01(color)) => {
                Some(Self {
                    link01: Some(color),
                    ..*self
                })
            }
            (Self { link12: None, ..}, Color12(color)) => {
                Some(Self {
                    link12: Some(color),
                    ..*self
                })
            }
            (_, _) => { None }
        }
    }
    fn claim_node0(&self, player: Player) -> Option<Self> {
        self.node0.is_none().then_some(
            Self {
                node0: Some(player),
                .. *self
            }
        )
    }
    fn claim_node1(&self, player: Player) -> Self {
        Self {
            node1: Some(player),
            .. *self
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct State {
    next: Player,
    network: Network,
}

impl State {
    // fn set_node0_claim(mut self) -> Self {
    //
    // }
    // fn next(self) -> impl Iterator<Item = (State, Action)> {
    //     match self.next {
    //         Player::Red => {
    //             match self.network {
    //                 Network { node0: None, .. } => self.network.claim_node0(self.next)
    //             }
    //         }
    //         Player::Blue => {
    //
    //         }
    //     }
    // }
}

#[derive(Debug, Default)]
struct Tangled {
    state: State,
}
