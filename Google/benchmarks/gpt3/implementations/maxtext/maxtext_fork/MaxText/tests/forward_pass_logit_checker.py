#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


# This forward_pass_logit_checker.py file compares the logits generated by MaxText implementation for some input prompts
# with the golden logits for those input prompts for a particular model. This forward_pass_logit_checker.py is generic that
# it can work with different models and expects an input file called golden_data_<model_name>.jsonl to be present
# under MaxText/test_assets
# For e.g., MaxText/test_assets/golden_data_llama2-7b.jsonl
# The golden jsonl file is a simple jsonlines file with each line is in the format of a dictionary containing the following
# required keys:
# 1. prompt: A string representing the prompt, for e.g., "I love to",
# 2. tokens: token ids after tokenizing the prompt,
# 3. logits: golden logits meaning the ideal logits generated by the model in question when fed with the prompt in #1
# There can be multiple such test cases in the jsonl file, each test case is a new line in the jsonl file
# This forward_pass_logit_checker.py runs the forward pass with the input tokens and asserts that the logits generated by the
# MaxText implementation of the same model matches the golden logits closely
# Users could use a script similar to MaxText/scratch_code/golden_llama2-7b_export.ipynb to create this jsonl file

"""Check if the logits generated by a model's MaxText implementation matches golden logits for the same inputs"""
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
maxtext_parent_dir = os.path.dirname(current_dir)
sys.path.append(maxtext_parent_dir)

import max_logging

max_logging.log(f"Added parent directory = {maxtext_parent_dir}")

import common_types
import jax
import jax.numpy as jnp
import numpy as np
import pyconfig
import jsonlines
import train


def get_data(golden_data, golden_data_index, config):
  """Get the golden data for the test indexed at golden_data_index"""

  max_logging.log(f"Comparing forward pass for golden data index = {golden_data_index} ")
  max_logging.log(f"config.global_batch_size_to_train_on={config.global_batch_size_to_train_on}")
  s = (config.global_batch_size_to_train_on, config.max_target_length)
  ids = np.asarray(golden_data[golden_data_index]["tokens"], dtype=np.int32)

  logits = np.asarray(golden_data[golden_data_index]["logits"], dtype=np.float32)
  max_logging.log(f" prompt=\"{golden_data[golden_data_index]['prompt']}\" raw ids={ids}, logits.shape = {logits.shape}")

  decoder_segment_ids = jax.numpy.zeros(s) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
  decoder_positions = jnp.stack(
      [jnp.arange(config.max_target_length, dtype=jnp.int32) for _ in range(config.global_batch_size_to_train_on)]
  )

  ids = jnp.stack([ids for _ in range(config.global_batch_size_to_train_on)])
  max_logging.log(f"ids={ids}, decoder_segment_ids = {decoder_segment_ids}, decoder_positions= {decoder_positions}")

  return ids, decoder_segment_ids, decoder_positions, logits


def main(config, test_args):
  """Test the Whole Model of model_name"""

  # initialize the model with weights from reference ckpt
  (
      init_rng,
      _,
      _,
      _,
      model,
      _,
      _,
      _,
      _,
      state,
  ) = train.setup_train_loop(config)

  input_golden_data_path = "MaxText/test_assets/golden_data_" + config.model_name + ".jsonl"
  with jsonlines.open(input_golden_data_path, "r") as f:
    golden_data = list(f)

  for golden_data_index in range(len(golden_data)):
    ids, decoder_segment_ids, decoder_positions, golden_logits = get_data(golden_data, golden_data_index, config)

    full_train_logits = model.apply(
        state.params,
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False,
        rngs={"aqt": init_rng},
    )
    full_train_logits = jax.experimental.multihost_utils.process_allgather(full_train_logits)
    max_logging.log(f"{golden_logits[0]=}")
    max_logging.log(f"{full_train_logits[0, 0, :]=}")
    token_size = int(test_args.token_size) if test_args.token_size else golden_logits.shape[0]
    max_logging.log(
        f"Max Numerical Difference {np.max(np.subtract(full_train_logits[0, :token_size, :], golden_logits[:token_size, :]))}"
    )

    model_probabilities = jax.nn.softmax(full_train_logits[0, :token_size, :], axis=-1)
    golden_probabilities = jax.nn.softmax(golden_logits[:token_size, :], axis=-1)

    max_logging.log(f"{golden_probabilities[0]=}")
    max_logging.log(f"{model_probabilities[0]=}")

    kl_div = jax.numpy.sum(jax.scipy.special.kl_div(golden_probabilities, model_probabilities), axis=-1)
    max_logging.log(f"KL divergence = {kl_div}, max KL divergence = {jax.numpy.max(kl_div)}")

    if test_args.max_kl_div is not None:
      max_logging.log("Checking KL Divergence between train distribution and golden distribution")
      assert jax.numpy.all(
          kl_div < test_args.max_kl_div
      ), f"KL divergence values exceed the specified threshold of {test_args.max_kl_div}. Max divergence: {jax.numpy.max(kl_div)}"
    else:
      max_logging.log("Checking Numerical Differences between train logits and golden logits")
      assert jax.numpy.allclose(
          full_train_logits[0, :token_size, :],
          golden_logits[:token_size, :],
          rtol=float(test_args.rtol),
          atol=float(test_args.atol),
          equal_nan=False,
      ), f"Logits do not match closely enough. Required rtol={test_args.rtol}, atol={test_args.atol}."


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  parser = argparse.ArgumentParser()
  parser.add_argument("--atol", type=float, required=False, default=0.1)
  parser.add_argument("--rtol", type=float, required=False, default=0.1)
  parser.add_argument("--token_size", type=int, required=False)
  parser.add_argument("--max_kl_div", type=float, required=False, default=None)
  test_args, _ = parser.parse_known_args()

  # Remove args defined in this test file to avoid error from pyconfig
  model_args = sys.argv
  to_remove_args = ["--atol", "--rtol", "--token_size", "--max_kl_div"]
  for arg in to_remove_args:
    model_args = [s for s in model_args if not s.startswith(arg)]

  pyconfig.initialize(model_args)
  cfg = pyconfig.config
  main(cfg, test_args)
