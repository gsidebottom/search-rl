use crate::env::{Action, ActionMap, State};
use rand::random_range;
use smallvec::Array;
use std::cell::OnceCell;
use std::default::Default;
use std::ops::{Index, IndexMut};
use itertools::Itertools;

#[derive(Debug, Clone, Copy)]
pub struct NodeRef(usize);

impl NodeRef {
    fn index(&self) -> usize {
        self.0
    }
}

#[derive(Default)]
pub struct Nodes<const N: usize, S: State>(Vec<Node<N, S>>);

impl<const N: usize, S: State> Nodes<N, S> {
    pub fn new() -> (NodeRef, Self) {
        let mut nodes = Self(Vec::new());
        let root_ref = nodes.add_node(S::init());
        (root_ref, nodes)
    }

    /// select an action taking to child node/state
    pub fn select_action(&mut self, node_ref: NodeRef) -> (Action, NodeRef) {
        // lazily create takes and children/states
        if !self[node_ref].has_takes() {
            self.add_takes(node_ref);
            self.add_children(node_ref)
        }
        self[node_ref].select_action()
    }

    fn add_node(&mut self, state: S) -> NodeRef {
        let index = self.0.len();
        self.0.push(Node::new(state));
        NodeRef(index)
    }

    fn add_takes(&self, node_ref: NodeRef) {
        let mut last_node_ref = self.0.len()-1;
        self[node_ref].add_takes(|| { last_node_ref += 1; NodeRef(last_node_ref) });
    }

    fn add_children(&mut self, node_ref: NodeRef) {
        let state = &self[node_ref].state;
        let child_states = state.action_iter().map(|action| state.take(action)).collect_vec();
        child_states.into_iter().for_each(|child_state| {
            _ = self.add_node(child_state);
        })
    }
}

impl<const N: usize, S: State> Index<NodeRef> for Nodes<N, S> {
    type Output = Node<N, S>;

    fn index(&self, index: NodeRef) -> &Self::Output {
        &self.0[index.index()]
    }
}

impl<const N: usize, S: State> IndexMut<NodeRef> for Nodes<N, S> {
    fn index_mut(&mut self, index: NodeRef) -> &mut Self::Output {
        &mut self.0[index.index()]
    }
}

/// result of taking an action
pub struct Take<const N: usize> {
    /// the action taken
    action: Action,
    /// the child node in the search tree resulting from taking the action
    child_ref: NodeRef,
    /// N(s, a): The number of times action a has been taken from state s
    count: usize,
    /// W(s, a): The total summed value V obtained from every time we’ve taken action a from
    /// state s in the current search tree
    total_value: f32,
    /// P(s, a): The prior probability of selecting action a from state s; this is the output of
    /// the neural net on state s, except for when s is a root node, where we add Dirichlet
    /// noise (I’ll explain this later) to the neural net output to encourage exploration from the
    /// root node
    prior: f32,
}

impl<const N: usize> Take<N> {
    /// Q(s, a): The mean value V obtained from taking action a from state s, equal to W(s, a) / N(s, a)
    pub fn quality(&self) -> f32 {
        self.total_value / self.count as f32
    }
}

unsafe impl<const N: usize> Array for Take<N> {
    type Item = Take<N>;

    fn size() -> usize {
        N
    }
}

pub struct Node<const N: usize, S: State> {
    state: S,
    /// N(s): The number of times state s has been visited
    visit_count: usize,
    reward: OnceCell<Option<f32>>,
    takes: OnceCell<ActionMap<Take<N>>>,
}

impl<const N: usize, S: State> Node<N, S> {
    pub fn new(state: S) -> Self {
        Self {
            state,
            visit_count: 0,
            reward: OnceCell::new(),
            takes: OnceCell::new(),
        }
    }

    /// the number of times this state has been visited
    pub fn visit_count(&self) -> usize {
        self.visit_count
    }

    /// the reward for this state, none if non-terminal state
    pub fn reward(&self) -> &Option<f32> {
        self.reward.get_or_init(|| self.state.reward())
    }

    pub fn has_takes(&self) -> bool {
        self.takes.get().is_some()
    }

    /// add the takes
    pub fn add_takes(&self, mut new_node_ref: impl FnMut() -> NodeRef) {
        self.takes.get_or_init(|| {
            let prior = 1.0 / self.state.action_count() as f32;
            let action_result_iter = self.state.action_iter().map(|a| {
                let child_ref = new_node_ref();
                Take {
                    action: a,
                    child_ref,
                    count: 0,
                    total_value: 0.0,
                    prior,
                }
            });
            ActionMap::new(action_result_iter)
        });
    }

    /// get the action takes
    fn takes(&self) -> &ActionMap<Take<N>> {
        self.takes.get().expect("action takes should be created")
    }

    /// select the action to take
    pub fn select_action(&self) -> (Action, NodeRef) {
        let r: f32 = random_range(0.0..=1.0);
        self.takes()
            .iter()
            .scan(
                0.0,
                |cumulative,
                 Take {
                     action,
                     child_ref,
                     prior,
                     ..
                 }| {
                    *cumulative += *prior;
                    (*cumulative <= r).then_some((*action, *child_ref))
                },
            )
            .last()
            .expect("at least one action")
    }
}

pub struct Tree<const N: usize, S: State> {
    root_ref: NodeRef,
    nodes: Nodes<N, S>,
}

impl<const N: usize, S: State> Default for Tree<N, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, S: State> Tree<N, S> {
    pub fn new() -> Self {
        let (root_ref, nodes) = Nodes::new();
        Self { root_ref, nodes }
    }

    pub fn simulate(&mut self, count: usize) {
        let mut back = Vec::with_capacity(128);
        for _ in 0..count {
            let mut curr_ref = self.root_ref;
            while self.nodes[curr_ref].visit_count() > 0
                && self.nodes[curr_ref].reward().is_none()
            {
                let (taken, new_curr_ref) = self.nodes.select_action(curr_ref);
                back.push((curr_ref, taken));
                curr_ref = new_curr_ref;
            }
            let mut value = match self.nodes[curr_ref].reward() {
                Some(reward) => {
                    // If it is a terminal state, adjudicate it, and set the value V(s) to the game
                    // outcome (player one wins = +1, draw = +0.01, player two wins = -1) for a
                    // two-player game with player one taking action a — (draws get assigned a
                    // small positive value); recall that V(s) => -V(s) for the competing agent
                    // (player two’s values are the negative of player one’s values) if your
                    // situation is a zero-sum game
                    *reward
                }
                None => {
                    // If you have not visited it yet, and it’s not a terminal state, call this a
                    // leaf state. Send the leaf state into the neural net, and initialize the
                    // state’s prior probabilities P(s, a) and value V(s) to whatever the neural net
                    // outputs, adding Dirichlet noise to P(s, a) if s is a root state.
                    // first visit of non-terminal
                    0.0
                }
            };
            // Back-propagate the value V(s) of the leaf or terminal state to the state that
            // led to it, being careful with minus signs; increment the number of times each
            // state-action pair N(s, a) was visited along the path from the root node to the
            // leaf or terminal node; adjust N(s), W(s, a), and Q(s, a) for all the states and
            // actions along this path
            while let Some((prev_ref, taken)) = back.pop() {
                let prev = &mut self.nodes[prev_ref];
                prev.visit_count += 1;
                value = prev.state.value(taken, value);
                let prev_action_result = &mut prev
                    .takes
                    .get_mut()
                    .expect("action results should be expanded")[taken];
                prev_action_result.count += 1;
                prev_action_result.total_value += value
            }
        }
    }
}
