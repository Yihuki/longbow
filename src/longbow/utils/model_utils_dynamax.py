"""Dynamax-based HMM model construction utilities for Longbow.

This module provides ModelBuilder class with methods to construct various HMM models
using dynamax's CategoricalHMM for sequence modeling tasks.
"""

import logging
import re

import jax.numpy as jnp
import jax.random as jr

from dynamax.hidden_markov_model import CategoricalHMM

import longbow.utils.constants
from .constants import RANDOM_SEGMENT_NAME, FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME, HPR_SEGMENT_TYPE_NAME, \
    RANDOM_SILENT_STATE_A, RANDOM_SILENT_STATE_B, RANDOM_BASE_STATE

logging.basicConfig(stream=__import__('sys').stderr)
logger = logging.getLogger(__name__)


class ModelBuilder:
    """Utilities for constructing a full Longbow model using dynamax CategoricalHMM."""

    # Define constants for all our default probabilities here:
    RANDOM_BASE_PROB = 0.25

    PER_BASE_MATCH_PROB = 0.94
    PER_BASE_MISMATCH_PROB = 0.02

    MATCH_MATCH_PROB = 0.90
    MATCH_INDEL_PROB = 0.05
    MATCH_TRAIL_INSERT_PROB = 0.90

    INDEL_CONTINUATION_PROB = 0.7
    INDEL_SWITCH_PROB = 0.3

    START_RANDOM_PROB = 1.0

    START_AND_END_RANDOM_PROB = 0.5
    RAND_RAND_PROB = 0.5
    RAND_INS_CONTINUATION_PROB = 0.8
    RAND_INS_TO_DEL_PROB = 0.1
    RAND_INS_END_PROB = 0.1

    NAMED_RAND_CONTINUE_PROB = 0.9
    NAMED_RAND_EXIT_PROB = 1 - NAMED_RAND_CONTINUE_PROB

    HPR_MATCH_PROB = 0.9
    HPR_MISMATCH_PROB = (1 - HPR_MATCH_PROB) / 3

    HPR_BOOKEND_MATCH_PROB = 0.99
    HPR_BOOKEND_MISMATCH_PROB = (1 - HPR_BOOKEND_MATCH_PROB) / 3

    HPR_SUDDEN_END_PROB = 0.01
    HPR_MODEL_RECURRENCE_PROB = 0.1

    SUDDEN_END_PROB = 0.01
    MATCH_END_PROB = 0.1

    # Random state transition probabilities
    RAND_TO_CDNA_START_PROB = 0.5
    RAND_TO_ARRAY_START_PROB = 0.5
    ARRAY_END_TO_RANDOM_PROB = 0.01
    ARRAY_END_TO_CDNA_PROB = 1.0 - 0.01 - 0.01  # 0.98, but we use 1.0 in actual implementation
    CDNA_END_TO_RANDOM_PROB = 0.01
    CDNA_END_TO_ARRAY_PROB = 1.0 - 0.01  # 0.99

    # Nucleotide mapping for emission probabilities
    NUCLEOTIDES = ['A', 'C', 'G', 'T']
    NUM_CLASSES = 4

    @staticmethod
    def _get_nuc_idx(nuc):
        """Get the index for a nucleotide."""
        return ModelBuilder.NUCLEOTIDES.index(nuc)

    @staticmethod
    def _create_emission_probs(match_nuc=None, pseudocount=1e-10):
        """Create emission probability matrix.

        Args:
            match_nuc: The nucleotide that should match (for match states).
                      If None, use uniform random distribution.
            pseudocount: Small value to add for numerical stability.

        Returns:
            jax array of shape (4,) with emission probabilities
        """
        if match_nuc is not None:
            # Use PER_BASE_MATCH_PROB and PER_BASE_MISMATCH_PROB to match pomegranate
            # Apply pseudocount for numerical stability while maintaining relative probabilities
            probs = jnp.array([ModelBuilder.PER_BASE_MISMATCH_PROB] * 4)
            idx = ModelBuilder._get_nuc_idx(match_nuc)
            probs = probs.at[idx].set(ModelBuilder.PER_BASE_MATCH_PROB)
        else:
            # Use uniform random distribution for insert/delete states
            probs = jnp.array([ModelBuilder.RANDOM_BASE_PROB] * 4)
        return probs

    @staticmethod
    def make_global_alignment_model(seq, name=None, pseudocount=1e-10):
        """Build a global alignment model for a known sequence.
        
        NOTE: In dynamax, there's no silent state support. So we use M->M skip
        to simulate deletions instead of using D states.
        """
        if name is None:
            name = "GLOBAL_ALIGNMENT"

        logger.debug("Making Model: GLOBAL_ALIGNMENT (%s)", name)

        seq_len = len(seq)
        num_states = 1 + seq_len * 2  # I0 + 2 states per base position (M + I), no D states

        state_names = [f"{name}:I0"]
        for i in range(seq_len):
            state_names.extend([
                f"{name}:M{i + 1}",
                f"{name}:I{i + 1}"
            ])

        state_to_idx = {name: idx for idx, name in enumerate(state_names)}
        emission_probs = jnp.zeros((num_states, 1, ModelBuilder.NUM_CLASSES))

        # I0 state: uniform random
        emission_probs = emission_probs.at[0, 0, :].set(
            ModelBuilder._create_emission_probs(pseudocount=pseudocount)
        )

        # Per-position states (no D states in dynamax)
        for i in range(seq_len):
            m_idx = state_to_idx[f"{name}:M{i + 1}"]
            i_idx = state_to_idx[f"{name}:I{i + 1}"]

            # M states: emit the expected nucleotide
            emission_probs = emission_probs.at[m_idx, 0, :].set(
                ModelBuilder._create_emission_probs(seq[i], pseudocount=pseudocount)
            )
            # I states: emit random nucleotide
            emission_probs = emission_probs.at[i_idx, 0, :].set(
                ModelBuilder._create_emission_probs(pseudocount=pseudocount)
            )

        initial_probs = jnp.zeros(num_states)
        # I0 gets some probability (for insertions at start)
        initial_probs = initial_probs.at[0].set(ModelBuilder.MATCH_INDEL_PROB)
        # M1 gets most of the probability (for normal matching)
        initial_probs = initial_probs.at[state_to_idx[f"{name}:M1"]].set(ModelBuilder.MATCH_MATCH_PROB)

        trans_matrix = jnp.zeros((num_states, num_states))
        trans_matrix = trans_matrix.at[0, state_to_idx[f"{name}:I0"]].set(1.0)

        # I0 transitions: Following pomegranate's behavior
        # I0->I0 = INDEL_CONTINUATION_PROB = 0.7
        # I0->M1 = INDEL_SWITCH_PROB = 0.3 (no D states, so all insert prob goes to M1)
        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:I0"], state_to_idx[f"{name}:I0"]
        ].set(ModelBuilder.INDEL_CONTINUATION_PROB)
        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:I0"], state_to_idx[f"{name}:M1"]
        ].set(ModelBuilder.INDEL_SWITCH_PROB)

        # Per-position transitions (no D states in dynamax)
        # M state transitions: M -> I (insert), M -> M (match), M -> skip to next M (deletion)
        # I state transitions: I -> I (continue insert), I -> M (end insert)
        for c in range(1, seq_len):
            m_prev_idx = state_to_idx[f"{name}:M{c}"]
            m_curr_idx = state_to_idx[f"{name}:M{c + 1}"]
            i_prev_idx = state_to_idx[f"{name}:I{c}"]
            i_curr_idx = state_to_idx[f"{name}:I{c + 1}"]
            
            # I{c} transitions
            trans_matrix = trans_matrix.at[i_prev_idx, i_curr_idx].set(ModelBuilder.INDEL_SWITCH_PROB / 2)
            trans_matrix = trans_matrix.at[i_prev_idx, m_curr_idx].set(ModelBuilder.INDEL_CONTINUATION_PROB)

            # M{c} transitions: M->I (0.05), M->M (0.90), M->M_skip (0.05) for deletion
            trans_matrix = trans_matrix.at[m_prev_idx, i_curr_idx].set(ModelBuilder.MATCH_INDEL_PROB)
            trans_matrix = trans_matrix.at[m_prev_idx, m_curr_idx].set(ModelBuilder.MATCH_MATCH_PROB + ModelBuilder.MATCH_INDEL_PROB)

        # Final state transitions
        # For the last I state: I_last -> I_last (1.0) - stays in insert state until sequence ends
        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:I{seq_len}"], state_to_idx[f"{name}:I{seq_len}"]
        ].set(1.0)

        # For the last M state: M_last -> I_last (0.05) + M_last -> M_last (0.95) for deletion simulation
        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:M{seq_len}"], state_to_idx[f"{name}:I{seq_len}"]
        ].set(ModelBuilder.MATCH_INDEL_PROB)
        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:M{seq_len}"], state_to_idx[f"{name}:M{seq_len}"]
        ].set(1.0 - ModelBuilder.MATCH_INDEL_PROB)

        from .model_dynamax import LibraryModel
        model = LibraryModel(n_states=num_states, n_emissions=ModelBuilder.NUM_CLASSES,
                            state_names=state_names)
        model.build(states=state_names, initial_probs=initial_probs,
                   trans_matrix=trans_matrix, emissions_matrix=emission_probs.squeeze(1))

        return model

    @staticmethod
    def make_homopolymer_repeat_model(name, nucleotide, expected_length, pseudocount=1e-10):
        """Build a homopolymer repeat model."""
        logger.debug("Making Model: HOMOPOLYMER_REPEAT (%s:%s x %d)", name, nucleotide, expected_length)

        num_states = 1 + expected_length * 2
        state_names = [f"{name}:I0"]
        for i in range(expected_length):
            state_names.extend([f"{name}:M{i + 1}", f"{name}:I{i + 1}"])

        state_to_idx = {name: idx for idx, name in enumerate(state_names)}
        emission_probs = jnp.zeros((num_states, 1, ModelBuilder.NUM_CLASSES))

        emission_probs = emission_probs.at[0, 0, :].set(
            ModelBuilder._create_emission_probs(pseudocount=pseudocount)
        )

        for i in range(expected_length):
            m_idx = state_to_idx[f"{name}:M{i + 1}"]
            i_idx = state_to_idx[f"{name}:I{i + 1}"]

            if i == 0 or i == expected_length - 1:
                match_prob = ModelBuilder.HPR_BOOKEND_MATCH_PROB
                mismatch_prob = ModelBuilder.HPR_BOOKEND_MISMATCH_PROB
            else:
                match_prob = ModelBuilder.HPR_MATCH_PROB
                mismatch_prob = ModelBuilder.HPR_MISMATCH_PROB

            m_probs = jnp.array([pseudocount] * 4)
            nuc_idx = ModelBuilder._get_nuc_idx(nucleotide)
            m_probs = m_probs.at[nuc_idx].set(match_prob - 3 * pseudocount)
            for j in range(4):
                if j != nuc_idx:
                    m_probs = m_probs.at[j].set(mismatch_prob)
            emission_probs = emission_probs.at[m_idx, 0, :].set(m_probs)

            emission_probs = emission_probs.at[i_idx, 0, :].set(
                ModelBuilder._create_emission_probs(pseudocount=pseudocount)
            )

        initial_probs = jnp.zeros(num_states)
        initial_probs = initial_probs.at[0].set(ModelBuilder.MATCH_INDEL_PROB)
        initial_probs = initial_probs.at[state_to_idx[f"{name}:M1"]].set(ModelBuilder.MATCH_MATCH_PROB)

        trans_matrix = jnp.zeros((num_states, num_states))
        # I0 transitions: Following pomegranate's behavior
        # I0->I0 = INDEL_CONTINUATION_PROB = 0.7
        # I0->M1 = INDEL_SWITCH_PROB / 2 = 0.15
        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:I0"], state_to_idx[f"{name}:I0"]
        ].set(ModelBuilder.INDEL_CONTINUATION_PROB)
        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:I0"], state_to_idx[f"{name}:M1"]
        ].set(ModelBuilder.INDEL_SWITCH_PROB / 2)

        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:M1"], num_states - 1
        ].set(ModelBuilder.HPR_SUDDEN_END_PROB)

        for c in range(1, expected_length):
            trans_matrix = trans_matrix.at[
                state_to_idx[f"{name}:I{c}"], state_to_idx[f"{name}:I{c}"]
            ].set(ModelBuilder.INDEL_SWITCH_PROB / 2)
            trans_matrix = trans_matrix.at[
                state_to_idx[f"{name}:I{c}"], state_to_idx[f"{name}:M{c + 1}"]
            ].set(ModelBuilder.INDEL_CONTINUATION_PROB)

            trans_matrix = trans_matrix.at[
                state_to_idx[f"{name}:M{c}"], state_to_idx[f"{name}:I{c}"]
            ].set(ModelBuilder.MATCH_INDEL_PROB)
            trans_matrix = trans_matrix.at[
                state_to_idx[f"{name}:M{c}"], state_to_idx[f"{name}:M{c + 1}"]
            ].set(ModelBuilder.MATCH_MATCH_PROB)

        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:I{expected_length}"], state_to_idx[f"{name}:I{expected_length}"]
        ].set(ModelBuilder.INDEL_SWITCH_PROB / 2)

        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:M{expected_length}"], state_to_idx[f"{name}:I{expected_length}"]
        ].set(ModelBuilder.MATCH_TRAIL_INSERT_PROB)
        trans_matrix = trans_matrix.at[
            state_to_idx[f"{name}:M{expected_length}"], num_states - 1
        ].set(ModelBuilder.MATCH_END_PROB)

        from .model_dynamax import LibraryModel
        model = LibraryModel(n_states=num_states, n_emissions=ModelBuilder.NUM_CLASSES,
                            state_names=state_names)
        model.build(states=state_names, initial_probs=initial_probs,
                   trans_matrix=trans_matrix, emissions_matrix=emission_probs.squeeze(1))

        return model

    @staticmethod
    def make_named_random_model(name, pseudocount=1e-10):
        """Build a named random segment model."""
        logger.debug("Making Model: NAMED RANDOM (%s)", name)
        state_name = f"{name}:{RANDOM_BASE_STATE}"
        return [state_name]

    @staticmethod
    def make_random_repeat_model(pseudocount=1e-10):
        """Build a random repeat model."""
        logger.debug("Making Model: RANDOM REPEAT")
        state_names = [
            f"{RANDOM_SEGMENT_NAME}:{RANDOM_BASE_STATE}",
            f"{RANDOM_SEGMENT_NAME}:{RANDOM_SILENT_STATE_A}",
            f"{RANDOM_SEGMENT_NAME}:{RANDOM_SILENT_STATE_B}"
        ]
        return state_names

    @staticmethod
    def make_fixed_length_random_segment(name, length, pseudocount=1e-10):
        """Build a fixed length random segment model."""
        logger.debug("Making Model: FIXED_LENGTH_RANDOM (%s:%d)", name, length)
        state_names = [f"{name}:M{i + 1}" for i in range(length)]
        return state_names

    @staticmethod
    def make_full_longbow_model(array_model, cdna_model, model_name=None, pseudocount=1e-10):
        """Build a complete Longbow model combining array and cdna models."""
        if model_name is None:
            model_name = f"{array_model['name']}+{cdna_model['name']}"

        logger.info("Building full Longbow model: %s", model_name)

        # Collect all segments
        all_segments = []
        segment_types = []
        segment_params = []

        # Add random states
        random_start_states = ModelBuilder.make_random_repeat_model(pseudocount=pseudocount)
        random_state_count = len(random_start_states)

        # Add array adapters
        for adapter_name in array_model['structure']:
            seq = array_model['adapters'][adapter_name]
            all_segments.append(adapter_name)
            segment_types.append('ga')
            segment_params.append({'seq': seq})

        # Add cdna segments
        named_random_segments = set(cdna_model.get('named_random_segments', []))

        for adapter_name in cdna_model['structure']:
            adapter_def = cdna_model['adapters'][adapter_name]

            if isinstance(adapter_def, str):
                if adapter_name in named_random_segments:
                    all_segments.append(adapter_name)
                    segment_types.append('named_rand')
                    segment_params.append({})
                else:
                    all_segments.append(adapter_name)
                    segment_types.append('ga')
                    segment_params.append({'seq': adapter_def})

            elif isinstance(adapter_def, dict):
                if FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME in adapter_def:
                    length = adapter_def[FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME]
                    all_segments.append(adapter_name)
                    segment_types.append('flr')
                    segment_params.append({'length': length})

                elif HPR_SEGMENT_TYPE_NAME in adapter_def:
                    base, length = adapter_def[HPR_SEGMENT_TYPE_NAME]
                    all_segments.append(adapter_name)
                    segment_types.append('hpr')
                    segment_params.append({'base': base, 'length': length})

        # Build state space
        all_state_names = []
        state_to_segment = {}
        segment_state_ranges = []

        for state in random_start_states:
            all_state_names.append(state)
            state_to_segment[state] = ('random', 0)

        for seg_idx, (seg_name, seg_type, seg_param) in enumerate(
            zip(all_segments, segment_types, segment_params)
        ):
            seg_start_idx = len(all_state_names)

            if seg_type == 'ga':
                seq_len = len(seg_param['seq'])
                # NOTE: In dynamax, no silent states. Use M->M skip instead of D states.
                # So we only have I0 + M + I states (2 per position instead of 3)
                seg_states = [f"{seg_name}:I0"]
                for i in range(seq_len):
                    seg_states.extend([
                        f"{seg_name}:M{i + 1}",
                        f"{seg_name}:I{i + 1}"
                    ])

            elif seg_type == 'hpr':
                length = seg_param['length']
                seg_states = [f"{seg_name}:I0"]
                for i in range(length):
                    seg_states.extend([f"{seg_name}:M{i + 1}", f"{seg_name}:I{i + 1}"])

            elif seg_type == 'flr':
                length = seg_param['length']
                seg_states = [f"{seg_name}:M{i + 1}" for i in range(length)]

            elif seg_type == 'named_rand':
                seg_states = [f"{seg_name}:{RANDOM_BASE_STATE}"]

            else:
                raise ValueError(f"Unknown segment type: {seg_type}")

            for state in seg_states:
                all_state_names.append(state)
                state_to_segment[state] = (seg_name, seg_idx)

            segment_state_ranges.append((seg_start_idx, len(all_state_names)))

        num_states = len(all_state_names)
        logger.debug("Total states in combined model: %d", num_states)

        # Create emission probabilities
        emission_probs = jnp.zeros((num_states, 1, ModelBuilder.NUM_CLASSES))

        for i in range(random_state_count):
            state_name = all_state_names[i]
            emission_probs = emission_probs.at[i, 0, :].set(
                ModelBuilder._create_emission_probs(pseudocount=pseudocount)
            )

        for seg_idx, (seg_name, seg_type, seg_param) in enumerate(
            zip(all_segments, segment_types, segment_params)
        ):
            start_idx, end_idx = segment_state_ranges[seg_idx]

            if seg_type == 'ga':
                seq = seg_param['seq']
                seq_len = len(seq)

                emission_probs = emission_probs.at[start_idx, 0, :].set(
                    ModelBuilder._create_emission_probs(pseudocount=pseudocount)
                )

                for i in range(seq_len):
                    # In dynamax, no D states. Only M and I states.
                    m_idx = start_idx + 1 + i * 2
                    i_idx = start_idx + 2 + i * 2

                    # M states: emit expected nucleotide
                    emission_probs = emission_probs.at[m_idx, 0, :].set(
                        ModelBuilder._create_emission_probs(seq[i], pseudocount=pseudocount)
                    )
                    # I states: emit random nucleotide
                    emission_probs = emission_probs.at[i_idx, 0, :].set(
                        ModelBuilder._create_emission_probs(pseudocount=pseudocount)
                    )

            elif seg_type == 'hpr':
                base = seg_param['base']
                length = seg_param['length']

                emission_probs = emission_probs.at[start_idx, 0, :].set(
                    ModelBuilder._create_emission_probs(pseudocount=pseudocount)
                )

                for i in range(length):
                    m_idx = start_idx + 1 + i * 2
                    i_idx = start_idx + 2 + i * 2

                    # IMPORTANT: Following pomegranate, HPR intermediate positions
                    # use PER_BASE_MATCH_PROB (0.94), not HPR_MATCH_PROB (0.9)
                    if i == 0 or i == length - 1:
                        match_prob = ModelBuilder.HPR_BOOKEND_MATCH_PROB
                        mismatch_prob = ModelBuilder.HPR_BOOKEND_MISMATCH_PROB
                    else:
                        match_prob = ModelBuilder.PER_BASE_MATCH_PROB
                        mismatch_prob = ModelBuilder.PER_BASE_MISMATCH_PROB

                    m_probs = jnp.array([mismatch_prob] * 4)
                    nuc_idx = ModelBuilder._get_nuc_idx(base)
                    m_probs = m_probs.at[nuc_idx].set(match_prob)
                    emission_probs = emission_probs.at[m_idx, 0, :].set(m_probs)

                    emission_probs = emission_probs.at[i_idx, 0, :].set(
                        ModelBuilder._create_emission_probs(pseudocount=pseudocount)
                    )

            elif seg_type == 'flr':
                length = seg_param['length']
                for i in range(length):
                    m_idx = start_idx + i
                    emission_probs = emission_probs.at[m_idx, 0, :].set(
                        ModelBuilder._create_emission_probs(pseudocount=pseudocount)
                    )

            elif seg_type == 'named_rand':
                emission_probs = emission_probs.at[start_idx, 0, :].set(
                    ModelBuilder._create_emission_probs(pseudocount=pseudocount)
                )

        # Create initial probability distribution
        # NOTE: Following pomegranate's approach, we give equal initial probability
        # to both the first array adapter and first cDNA adapter
        # This allows Viterbi to choose based on sequence match quality
        
        # Calculate first array and first cDNA adapter indices
        first_array_idx = random_state_count
        cdna_ga_start = None
        for seg_idx, (seg_name, seg_type, seg_param) in enumerate(
            zip(all_segments, segment_types, segment_params)
        ):
            if seg_type == 'ga' and seg_name in cdna_model['structure']:
                cdna_ga_start = segment_state_ranges[seg_idx][0]
                break
        
        # =================================================================
        # Initial probability distribution
        #
        # IMPORTANT: In pomegranate, the start state is a silent state that gets normalized.
        # After normalization, each transition from start has probability 1.0 / num_targets.
        # For our model, start has 4 targets: random:RI, random:RDA, A-start, 5p_Adapter-start.
        # After normalization: each gets 0.25 probability.
        #
        # Then each -start state distributes to I0/D1/M1:
        # - A-start -> A:I0(0.05), A:D1(0.05), A:M1(0.90)
        # - 5p_Adapter-start -> 5p_Adapter:I0(0.05), 5p_Adapter:D1(0.05), 5p_Adapter:M1(0.90)
        #
        # So the effective initial probabilities (after normalization) are:
        # - A:I0 = 0.25 * 0.05 = 0.0125
        # - A:D1 = 0.25 * 0.05 = 0.0125
        # - A:M1 = 0.25 * 0.90 = 0.225
        # - 5p_Adapter:I0 = 0.25 * 0.05 = 0.0125
        # - 5p_Adapter:D1 = 0.25 * 0.05 = 0.0125
        # - 5p_Adapter:M1 = 0.25 * 0.90 = 0.225
        # - random:RI = 0.25 * 0.5 = 0.125 (from random:RI -> random:RDA transition)
        # - random:RDA = 0.25 * 0.5 + 0.125 = 0.25 (0.5 from RDA -> array + 0.125 from RI)
        #
        # Total = 0.0125 + 0.0125 + 0.225 + 0.0125 + 0.0125 + 0.225 + 0.125 + 0.25 = 0.875
        # Wait, this doesn't sum to 1.0. Let me recalculate...
        #
        # Actually, the random:RI and random:RDA states need to be handled separately.
        # The probability flow is:
        # - start -> random:RI (0.25) -> random:RDA (0.5) = 0.125 to random:RDA
        # - start -> random:RDA (0.25) -> arrays (0.5) = 0.125 to arrays
        # - start -> A-start (0.25) -> A:I0/D1/M1 = 0.025/0.025/0.225
        # - start -> 5p_Adapter-start (0.25) -> 5p_Adapter:I0/D1/M1 = 0.025/0.025/0.225
        #
        # So random:RI initial = 0.25 (direct from start)
        # And random:RDA gets probability from both start->RDA (0.25) and RI->RDA (0.125)
        # Actually no, that's not right either. Let me think again...
        #
        # The correct interpretation is:
        # - random:RI and random:RDA are reached from start with probability 0.25 each
        # - A and 5p_Adapter are also reached from start with probability 0.25 each
        # - Within the random:RI/RDA cluster, probability flows from RI to RDA
        # - Within A and 5p_Adapter, probability is distributed to I0/D1/M1
        #
        # So the effective initial probabilities are:
        # - random:RI = 0.25 (from start)
        # - random:RDA = 0.25 (from start)
        # - A:I0 = 0.25 * 0.05 = 0.0125
        # - A:D1 = 0.25 * 0.05 = 0.0125
        # - A:M1 = 0.25 * 0.90 = 0.225
        # - 5p_Adapter:I0 = 0.25 * 0.05 = 0.0125
        # - 5p_Adapter:D1 = 0.25 * 0.05 = 0.0125
        # - 5p_Adapter:M1 = 0.25 * 0.90 = 0.225
        #
        # Total = 0.25 + 0.25 + 0.0125 + 0.0125 + 0.225 + 0.0125 + 0.0125 + 0.225 = 1.0
        # =================================================================
        
        # Set initial probabilities to match pomegranate's normalized result
        initial_probs = jnp.zeros(num_states)
        
        # Set initial probabilities for random states (from start->random:RI and start->random:RDA)
        # In pomegranate, after normalization: start->random:RI = 0.25, start->random:RDA = 0.25
        initial_probs = initial_probs.at[0].set(0.25)  # random:RI
        initial_probs = initial_probs.at[1].set(0.25)  # random:RDA
        
        # Set initial probabilities for first array adapter (A path)
        # A:I0 = 0.025, A:M1 = 0.225 (no D states in dynamax)
        if first_array_idx < num_states:
            first_array_i0_idx = first_array_idx  # A:I0
            first_array_m1_idx = first_array_idx + 1  # A:M1 (no D1)
            initial_probs = initial_probs.at[first_array_i0_idx].set(0.025)  # A:I0
            initial_probs = initial_probs.at[first_array_m1_idx].set(0.225)  # A:M1
        
        # Set initial probabilities for first cDNA adapter (5p_Adapter path)
        # 5p_Adapter:I0 = 0.025, 5p_Adapter:M1 = 0.225 (no D states in dynamax)
        if cdna_ga_start is not None:
            cdna_i0_idx = cdna_ga_start
            cdna_m1_idx = cdna_ga_start + 1  # M1 (no D1)
            initial_probs = initial_probs.at[cdna_i0_idx].set(0.025)  # 5p_Adapter:I0
            initial_probs = initial_probs.at[cdna_m1_idx].set(0.225)  # 5p_Adapter:M1
        
        # Normalize initial probabilities to sum to 1.0
        total_prob = jnp.sum(initial_probs)
        if total_prob > 0:
            initial_probs = initial_probs / total_prob

        # Create transition matrix
        trans_matrix = jnp.zeros((num_states, num_states))

        # Random state (RDA) transitions
        # NOTE: In dynamax, there's no silent start state. To match pomegranate behavior
        # (where start->I0/D1/M1 with start being silent), we directly transition to M1
        # (and D1 for cDNA) to avoid I0 consuming a position.
        #
        # Following pomegranate's normalized approach:
        # - Array adapters: each gets 0.05 probability, but we transition to M1 instead of I0
        # - cDNA adapter: 0.05 probability to 5p_Adapter:M1
        RDA_TO_ADAPTER_PROB = 0.05
        RDA_TO_CDNA_PROB = 0.05
        
        # Calculate all adapter M1 indices (I0 + 1 = M1 for ga segments, no D1 in dynamax)
        array_adapter_m1_indices = []
        for arr_idx, arr_name in enumerate(array_model['structure']):
            if arr_idx == 0:
                continue  # Skip A - only reachable from initial probability
            for seg_idx, (seg_name, seg_type, seg_param) in enumerate(
                zip(all_segments, segment_types, segment_params)
            ):
                if seg_name == arr_name and seg_type == 'ga':
                    # M1 index = start_idx + 1 (I0 + 1 = M1, no D1)
                    m1_idx = segment_state_ranges[seg_idx][0] + 1
                    array_adapter_m1_indices.append(m1_idx)
                    break
        
        # RDA -> array adapters M1 (B-P) with transition to I0/M1 (no D1 in dynamax)
        # Following pomegranate: each adapter gets equal probability, distributed to I0/M1
        num_arrays = len(array_adapter_m1_indices)
        for arr_m1_idx in array_adapter_m1_indices:
            # Find the I0/M1 indices for this array adapter
            arr_i0_idx = arr_m1_idx - 1
            
            # RDA -> I0 = 0.05 * (1.0 / num_arrays)
            # RDA -> M1 = 0.95 * (1.0 / num_arrays) (all D1 prob goes to M1)
            arr_prob = 1.0 / num_arrays
            trans_matrix = trans_matrix.at[1, arr_i0_idx].set(0.05 * arr_prob)
            trans_matrix = trans_matrix.at[1, arr_m1_idx].set(0.95 * arr_prob)
        
        # RDA -> first cdna adapter M1 (5p_Adapter) with transition to I0/M1 (no D1 in dynamax)
        if cdna_ga_start is not None:
            cdna_i0_idx = cdna_ga_start
            cdna_m1_idx = cdna_ga_start + 1  # M1 (no D1)
            
            # RDA -> I0 = 0.05 * 0.05 = 0.0025
            # RDA -> M1 = 0.95 * 0.05 = 0.0475 (all D1 prob goes to M1)
            trans_matrix = trans_matrix.at[1, cdna_i0_idx].set(0.05 * RDA_TO_CDNA_PROB)
            trans_matrix = trans_matrix.at[1, cdna_m1_idx].set(0.95 * RDA_TO_CDNA_PROB)

        # Build segment transitions
        for seg_idx, (seg_name, seg_type, seg_param) in enumerate(
            zip(all_segments, segment_types, segment_params)
        ):
            start_idx, end_idx = segment_state_ranges[seg_idx]

            if seg_type == 'ga':
                seq_len = len(seg_param['seq'])

                for i in range(seq_len):
                    # In dynamax, no D states. Only M and I states.
                    # Index: I0=start_idx, M1=start_idx+1, I1=start_idx+2, M2=start_idx+3, etc.
                    m_idx = start_idx + 1 + i * 2
                    i_idx = start_idx + 2 + i * 2

                    if i == 0:
                        # I0 transitions: Following pomegranate's behavior
                        # I0->I0 = INDEL_CONTINUATION_PROB = 0.7
                        # I0->M1 = INDEL_SWITCH_PROB / 2 = 0.15 (partial D1 prob)
                        # I0->M2 = INDEL_SWITCH_PROB / 2 = 0.15 (D1 prob, skip M1 to M2)
                        trans_matrix = trans_matrix.at[start_idx, start_idx].set(ModelBuilder.INDEL_CONTINUATION_PROB)
                        trans_matrix = trans_matrix.at[start_idx, m_idx].set(ModelBuilder.INDEL_SWITCH_PROB / 2)
                        # I0 -> M2 (skip M1, simulating deletion of first base)
                        if seq_len >= 2:
                            m2_idx = start_idx + 3  # M2 is at start_idx + 1 + 1 * 2 = start_idx + 3
                            trans_matrix = trans_matrix.at[start_idx, m2_idx].set(ModelBuilder.INDEL_SWITCH_PROB / 2)
                    else:
                        i_prev = start_idx + 2 + (i - 1) * 2
                        m_prev = start_idx + 1 + (i - 1) * 2

                        # I{c-1} transitions
                        trans_matrix = trans_matrix.at[i_prev, i_idx].set(ModelBuilder.INDEL_SWITCH_PROB / 2)
                        trans_matrix = trans_matrix.at[i_prev, m_idx].set(ModelBuilder.INDEL_CONTINUATION_PROB)

                        # M{c-1} transitions: 
                        # M->I (0.05), M->M (0.90), M->skip to next M+1 (0.05) for deletion
                        trans_matrix = trans_matrix.at[m_prev, i_idx].set(ModelBuilder.MATCH_INDEL_PROB)
                        trans_matrix = trans_matrix.at[m_prev, m_idx].set(ModelBuilder.MATCH_MATCH_PROB)
                        # M{c-1} -> M{c+1} (skip M{c}, simulating deletion)
                        # This is the M->D transition in pomegranate
                        if i < seq_len - 1:
                            m_next_next = start_idx + 1 + (i + 1) * 2  # M{i+2}
                            trans_matrix = trans_matrix.at[m_prev, m_next_next].set(ModelBuilder.MATCH_INDEL_PROB)

                # Last state transitions for ga segment (no D states in dynamax)
                last_i = end_idx - 1
                last_m = end_idx - 2

                # CRITICAL: I_last must have high probability to exit!
                # Following pomegranate: exit_prob = INDEL_CONTINUATION_PROB + INDEL_SWITCH_PROB/2 = 0.85
                # I_last->I_last = 0.15 (INDEL_SWITCH_PROB/2)
                # I_last->end = 0.85 (INDEL_CONTINUATION_PROB + INDEL_SWITCH_PROB/2)
                trans_matrix = trans_matrix.at[last_i, last_i].set(ModelBuilder.INDEL_SWITCH_PROB / 2)
                trans_matrix = trans_matrix.at[last_i, num_states - 1].set(
                    ModelBuilder.INDEL_CONTINUATION_PROB + ModelBuilder.INDEL_SWITCH_PROB / 2
                )

                # M_last transitions: M_last->I_last (0.05), M_last->M_last (0.94), M_last->end (0.01)
                # Following pomegranate: MATCH_INDEL_PROB=0.05, MATCH_MATCH_PROB=0.90, MATCH_END_PROB=0.1
                # But we need to add M_last->skip simulation which would go to end (no more M states)
                trans_matrix = trans_matrix.at[last_m, last_i].set(ModelBuilder.MATCH_INDEL_PROB)
                trans_matrix = trans_matrix.at[last_m, last_m].set(ModelBuilder.MATCH_MATCH_PROB)
                trans_matrix = trans_matrix.at[last_m, num_states - 1].set(ModelBuilder.MATCH_END_PROB)

            elif seg_type == 'hpr':
                length = seg_param['length']

                # I0 transitions: Following pomegranate's behavior
                # I0->I0 = INDEL_CONTINUATION_PROB = 0.7
                # I0->M1 = INDEL_SWITCH_PROB / 2 = 0.15
                # I0->M2 = INDEL_SWITCH_PROB / 2 = 0.15 (D1 prob, skip M1)
                trans_matrix = trans_matrix.at[start_idx, start_idx].set(ModelBuilder.INDEL_CONTINUATION_PROB)
                trans_matrix = trans_matrix.at[start_idx, start_idx + 1].set(ModelBuilder.INDEL_SWITCH_PROB / 2)
                if length >= 2:
                    m2_idx = start_idx + 3  # M2 is at start_idx + 1 + 1 * 2
                    trans_matrix = trans_matrix.at[start_idx, m2_idx].set(ModelBuilder.INDEL_SWITCH_PROB / 2)

                for i in range(length):
                    m_idx = start_idx + 1 + i * 2
                    i_idx = start_idx + 2 + i * 2

                    if i == 0:
                        trans_matrix = trans_matrix.at[m_idx, num_states - 1].set(ModelBuilder.HPR_SUDDEN_END_PROB)
                    else:
                        prev_i = start_idx + 2 + (i - 1) * 2
                        prev_m = start_idx + 1 + (i - 1) * 2

                        trans_matrix = trans_matrix.at[prev_i, i_idx].set(ModelBuilder.INDEL_SWITCH_PROB / 2)
                        trans_matrix = trans_matrix.at[prev_i, m_idx].set(ModelBuilder.INDEL_CONTINUATION_PROB)

                        trans_matrix = trans_matrix.at[prev_m, i_idx].set(ModelBuilder.MATCH_INDEL_PROB)
                        trans_matrix = trans_matrix.at[prev_m, m_idx].set(ModelBuilder.MATCH_MATCH_PROB)
                        # M->skip to M+1 (simulating deletion)
                        if i < length - 1:
                            m_next_next = start_idx + 1 + (i + 1) * 2
                            trans_matrix = trans_matrix.at[prev_m, m_next_next].set(ModelBuilder.MATCH_INDEL_PROB)

                last_m = end_idx - 2
                last_i = end_idx - 1
                # M_last transitions: M_last->I_last (0.05), M_last->M_last (0.94), M_last->end (0.01)
                trans_matrix = trans_matrix.at[last_m, last_i].set(ModelBuilder.MATCH_INDEL_PROB)
                trans_matrix = trans_matrix.at[last_m, last_m].set(ModelBuilder.MATCH_MATCH_PROB)
                trans_matrix = trans_matrix.at[last_m, num_states - 1].set(ModelBuilder.MATCH_END_PROB)
                # CRITICAL: I_last must have high probability to exit!
                # Following pomegranate: exit_prob = INDEL_CONTINUATION_PROB + INDEL_SWITCH_PROB/2 = 0.85
                trans_matrix = trans_matrix.at[last_i, last_i].set(ModelBuilder.INDEL_SWITCH_PROB / 2)
                trans_matrix = trans_matrix.at[last_i, num_states - 1].set(
                    ModelBuilder.INDEL_CONTINUATION_PROB + ModelBuilder.INDEL_SWITCH_PROB / 2
                )

            elif seg_type == 'flr':
                length = seg_param['length']
                for i in range(length):
                    m_idx = start_idx + i
                    if i < length - 1:
                        next_m = start_idx + i + 1
                        trans_matrix = trans_matrix.at[m_idx, next_m].set(ModelBuilder.MATCH_MATCH_PROB)
                    else:
                        trans_matrix = trans_matrix.at[m_idx, num_states - 1].set(ModelBuilder.MATCH_END_PROB)

            elif seg_type == 'named_rand':
                trans_matrix = trans_matrix.at[start_idx, num_states - 1].set(ModelBuilder.NAMED_RAND_EXIT_PROB)
                trans_matrix = trans_matrix.at[start_idx, start_idx].set(ModelBuilder.NAMED_RAND_CONTINUE_PROB)

        # =====================================================
        # CRITICAL: Add cross-segment transitions (like pomegranate)
        # =====================================================
        
        # Find the first cDNA adapter's I0 state
        first_cdna_ga_start = None
        for seg_idx, (seg_name, seg_type, seg_param) in enumerate(
            zip(all_segments, segment_types, segment_params)
        ):
            if seg_type == 'ga' and seg_name in cdna_model['structure']:
                first_cdna_ga_start = segment_state_ranges[seg_idx][0]
                break

        # 1. Array adapter -> cDNA first adapter transitions
        # For each array adapter (except the last one), connect to cDNA first adapter
        # NOTE: Following pomegranate, the last state of each array adapter transitions
        # to 5p_Adapter:I0 with probability 1.0
        # NOTE: In dynamax, no D states, so only last_m and last_i transitions
        
        for arr_idx, arr_name in enumerate(array_model['structure']):
            # Find this array adapter's segment info
            for seg_idx, (seg_name, seg_type, seg_param) in enumerate(
                zip(all_segments, segment_types, segment_params)
            ):
                if seg_name == arr_name and seg_type == 'ga':
                    arr_start_idx, arr_end_idx = segment_state_ranges[seg_idx]
                    arr_seq_len = len(seg_param['seq'])
                    
                    # Get last state indices (no D states in dynamax)
                    last_i = arr_end_idx - 1
                    last_m = arr_end_idx - 2
                    
                    # Clear any existing segment-internal transitions for last states
                    # (These were set in the per-segment block above, but we need to override them
                    # for states that have cross-segment transitions)
                    # Clear last_i's self-loop and end transition
                    trans_matrix = trans_matrix.at[last_i, last_i].set(0.0)
                    trans_matrix = trans_matrix.at[last_i, num_states - 1].set(0.0)
                    # Clear last_m's self-loop and end transition
                    trans_matrix = trans_matrix.at[last_m, last_m].set(0.0)
                    trans_matrix = trans_matrix.at[last_m, num_states - 1].set(0.0)
                    
                    # Connect last M/I state to cDNA first adapter (except last array adapter)
                    # Following pomegranate: A:I16, A:M16 all transition to 5p_Adapter-start with prob 1.0
                    # In dynamax, we distribute this to I0/M1: I0=0.05, M1=0.95
                    if arr_name != array_model['structure'][-1] and first_cdna_ga_start is not None:
                        cdna_i0 = first_cdna_ga_start
                        cdna_m1 = first_cdna_ga_start + 1  # M1 (no D1 in dynamax)
                        
                        trans_matrix = trans_matrix.at[last_i, cdna_i0].set(0.05)
                        trans_matrix = trans_matrix.at[last_i, cdna_m1].set(0.95)
                        
                        trans_matrix = trans_matrix.at[last_m, cdna_i0].set(0.05)
                        trans_matrix = trans_matrix.at[last_m, cdna_m1].set(0.95)
                    
                    # For the LAST array adapter, DON'T connect to 5p_Adapter
                    # (only cDNA adapters can connect back to array adapters)
                    break

        # 2. cDNA last adapter -> array adapters transitions
        # Connect cDNA last adapter (3p_Adapter) to all array adapters
        cdna_last_adapter = cdna_model['structure'][-1]
        
        for seg_idx, (seg_name, seg_type, seg_param) in enumerate(
            zip(all_segments, segment_types, segment_params)
        ):
            if seg_name == cdna_last_adapter:
                cdna_start_idx, cdna_end_idx = segment_state_ranges[seg_idx]
                break
        else:
            cdna_start_idx, cdna_end_idx = segment_state_ranges[-1]

        # Find cdna adapter length for M/D/I states
        cdna_adapter_def = cdna_model['adapters'][cdna_last_adapter]
        if isinstance(cdna_adapter_def, str):
            cdna_seq_len = len(cdna_adapter_def)
        elif isinstance(cdna_adapter_def, dict):
            if HPR_SEGMENT_TYPE_NAME in cdna_adapter_def:
                cdna_seq_len = cdna_adapter_def[HPR_SEGMENT_TYPE_NAME][1]
            else:
                cdna_seq_len = cdna_adapter_def.get(FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME, 1)
        else:
            cdna_seq_len = 1

        # Calculate the last M/I states for cDNA last adapter (no D states in dynamax)
        if cdna_model['adapters'][cdna_last_adapter] == RANDOM_SEGMENT_NAME:
            # For random segments, use the state itself
            cdna_last_states = [cdna_start_idx]  # Single state for named random
        else:
            # No D states in dynamax, so only M and I
            cdna_last_m = cdna_start_idx + 1 + (cdna_seq_len - 1) * 2 if seg_type == 'ga' else cdna_start_idx + cdna_seq_len - 1
            cdna_last_i = cdna_start_idx + 2 + (cdna_seq_len - 1) * 2 if seg_type == 'ga' else cdna_start_idx + cdna_seq_len - 1
            cdna_last_states = [cdna_last_m, cdna_last_i]

        # Connect cDNA last adapter to all array adapters
        # NOTE: Following pomegranate, each array adapter gets 1.0/num_arrays probability
        # For mas_15: 16 array adapters, each gets 0.0625
        # We transition to M1 instead of I0 to match pomegranate's silent start behavior.
        array_adapter_m1_indices = []
        for seg_idx, (seg_name, seg_type, seg_param) in enumerate(
            zip(all_segments, segment_types, segment_params)
        ):
            if seg_type == 'ga' and seg_name in array_model['structure']:
                # M1 index = start_idx + 1 (I0 + 1 = M1, no D1 in dynamax)
                m1_idx = segment_state_ranges[seg_idx][0] + 1
                array_adapter_m1_indices.append(m1_idx)

        # Probability for each array adapter transition (following pomegranate)
        num_arrays = len(array_adapter_m1_indices)
        array_trans_prob = 1.0 / num_arrays

        for cdna_last_state in cdna_last_states:
            for arr_m1_idx in array_adapter_m1_indices:
                arr_i0_idx = arr_m1_idx - 1
                
                # Distribute probability to I0/M1 (no D1 in dynamax)
                trans_matrix = trans_matrix.at[cdna_last_state, arr_i0_idx].set(0.05 * array_trans_prob)
                trans_matrix = trans_matrix.at[cdna_last_state, arr_m1_idx].set(0.95 * array_trans_prob)

        # 3. cDNA internal transitions (non-last adapters)
        for i, cdna_adapter in enumerate(cdna_model['structure'][:-1]):
            next_adapter = cdna_model['structure'][i + 1]
            
            # Find this cDNA adapter's segment info
            for seg_idx, (seg_name, seg_type, seg_param) in enumerate(
                zip(all_segments, segment_types, segment_params)
            ):
                if seg_name == cdna_adapter:
                    cdna_curr_start, cdna_curr_end = segment_state_ranges[seg_idx]
                    break
            
            # Find next cDNA adapter's I0 state
            for next_seg_idx, (next_seg_name, next_seg_type, next_seg_param) in enumerate(
                zip(all_segments, segment_types, segment_params)
            ):
                if next_seg_name == next_adapter:
                    next_start = segment_state_ranges[next_seg_idx][0]
                    break
            
            # Connect last states of current adapter to next adapter
            # Following pomegranate's -start state behavior:
            # - start->I0 with weight 0.05 (for ga and hpr segments)
            # - start->D1 with weight 0.05 (for ga segments only)
            # - start->M1 with weight 0.90
            # NOTE: In dynamax, no D states, so we adjust probabilities accordingly
            if seg_type == 'ga':
                curr_seq_len = len(seg_param['seq'])
                # No D states in dynamax
                last_m = cdna_curr_start + 1 + (curr_seq_len - 1) * 2
                last_i = cdna_curr_start + 2 + (curr_seq_len - 1) * 2
                
                # Clear any existing segment-internal transitions for last states
                trans_matrix = trans_matrix.at[last_i, last_i].set(0.0)
                trans_matrix = trans_matrix.at[last_i, num_states - 1].set(0.0)
                trans_matrix = trans_matrix.at[last_m, last_m].set(0.0)
                trans_matrix = trans_matrix.at[last_m, num_states - 1].set(0.0)
                
                # Set transitions to next adapter based on its type
                if next_seg_type == 'ga':
                    # Next is ga segment: has I0, M1 (no D1 in dynamax)
                    next_i0 = next_start
                    next_m1 = next_start + 1
                    
                    trans_matrix = trans_matrix.at[last_m, next_i0].set(0.05)
                    trans_matrix = trans_matrix.at[last_m, next_m1].set(0.95)
                    
                    trans_matrix = trans_matrix.at[last_i, next_i0].set(0.05)
                    trans_matrix = trans_matrix.at[last_i, next_m1].set(0.95)
                elif next_seg_type == 'hpr':
                    # Next is hpr segment: has I0, M1 (no D1)
                    next_i0 = next_start
                    next_m1 = next_start + 2  # I0 + 2 = M1 for hpr segments
                    
                    trans_matrix = trans_matrix.at[last_m, next_i0].set(0.05)
                    trans_matrix = trans_matrix.at[last_m, next_m1].set(0.95)
                    
                    trans_matrix = trans_matrix.at[last_i, next_i0].set(0.05)
                    trans_matrix = trans_matrix.at[last_i, next_m1].set(0.95)
                elif next_seg_type == 'flr':
                    # Next is flr segment: only M1-Mn (no I0/D1)
                    # FLR M1 is at next_start
                    next_m1 = next_start
                    
                    trans_matrix = trans_matrix.at[last_m, next_m1].set(1.0)
                    trans_matrix = trans_matrix.at[last_i, next_m1].set(1.0)
                else:
                    # named_rand or other: single state
                    trans_matrix = trans_matrix.at[last_m, next_start].set(1.0)
                    trans_matrix = trans_matrix.at[last_i, next_start].set(1.0)
            elif seg_type == 'hpr':
                curr_seq_len = seg_param['length']
                last_m = cdna_curr_start + 1 + (curr_seq_len - 1) * 2
                last_i = cdna_curr_start + 2 + (curr_seq_len - 1) * 2
                
                # Clear any existing segment-internal transitions for last states
                trans_matrix = trans_matrix.at[last_i, last_i].set(0.0)
                trans_matrix = trans_matrix.at[last_i, num_states - 1].set(0.0)
                trans_matrix = trans_matrix.at[last_m, last_m].set(0.0)
                trans_matrix = trans_matrix.at[last_m, num_states - 1].set(0.0)
                
                # Set transitions to next adapter based on its type
                if next_seg_type == 'ga':
                    # Next is ga segment: has I0, M1 (no D1 in dynamax)
                    next_i0 = next_start
                    next_m1 = next_start + 1
                    
                    trans_matrix = trans_matrix.at[last_m, next_i0].set(0.05)
                    trans_matrix = trans_matrix.at[last_m, next_m1].set(0.95)
                    
                    trans_matrix = trans_matrix.at[last_i, next_i0].set(0.05)
                    trans_matrix = trans_matrix.at[last_i, next_m1].set(0.95)
                elif next_seg_type == 'hpr':
                    # Next is hpr segment: has I0, M1 (no D1)
                    next_i0 = next_start
                    next_m1 = next_start + 2
                    
                    trans_matrix = trans_matrix.at[last_m, next_i0].set(0.05)
                    trans_matrix = trans_matrix.at[last_m, next_m1].set(0.95)
                    
                    trans_matrix = trans_matrix.at[last_i, next_i0].set(0.05)
                    trans_matrix = trans_matrix.at[last_i, next_m1].set(0.95)
                elif next_seg_type == 'flr':
                    # Next is flr segment: only M1-Mn (no I0/D1)
                    next_m1 = next_start
                    
                    trans_matrix = trans_matrix.at[last_m, next_m1].set(1.0)
                    trans_matrix = trans_matrix.at[last_i, next_m1].set(1.0)
                else:
                    trans_matrix = trans_matrix.at[last_m, next_start].set(1.0)
                    trans_matrix = trans_matrix.at[last_i, next_start].set(1.0)
            elif seg_type == 'flr':
                curr_seq_len = seg_param['length']
                last_m = cdna_curr_start + curr_seq_len - 1
                
                # Clear any existing segment-internal transitions for last state
                trans_matrix = trans_matrix.at[last_m, num_states - 1].set(0.0)
                
                trans_matrix = trans_matrix.at[last_m, next_start].set(1.0)
            else:
                # named_rand or other single state
                trans_matrix = trans_matrix.at[cdna_curr_start, next_start].set(1.0)

        # Create and return the model
        from .model_dynamax import LibraryModel
        model = LibraryModel(n_states=num_states, n_emissions=ModelBuilder.NUM_CLASSES,
                            state_names=all_state_names)
        model.build(states=all_state_names, initial_probs=initial_probs,
                   trans_matrix=trans_matrix, emissions_matrix=emission_probs.squeeze(1))

        return model


# Pre-configured models
ModelBuilder.pre_configured_models = {
    'array': {
        "mas_16": {
            "description": "16-element MAS-ISO-seq array",
            "version": "3.0.0",
            "structure": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"],
            "adapters": {
                "A": "AGCTTACTTGTGAAGA",
                "B": "ACTTGTAAGCTGTCTA",
                "C": "ACTCTGTCAGGTCCGA",
                "D": "ACCTCCTCCTCCAGAA",
                "E": "AACCGGACACACTTAG",
                "F": "AGAGTCCAATTCGCAG",
                "G": "AATCAAGGCTTAACGG",
                "H": "ATGTTGAATCCTAGCG",
                "I": "AGTGCGTTGCGAATTG",
                "J": "AATTGCGTAGTTGGCC",
                "K": "ACACTTGGTCGCAATC",
                "L": "AGTAAGCCTTCGTGTC",
                "M": "ACCTAGATCAGAGCCT",
                "N": "AGGTATGCCGGTTAAG",
                "O": "AAGTCACCGGCACCTT",
                "P": "ATGAAGTGGCTCGAGA",
                "Q": "AGTAGCTGTGTGCA",
            },
            "deprecated": False,
            "name": "mas_16",
        },
        "mas_15": {
            "description": "15-element MAS-ISO-seq array",
            "version": "3.0.0",
            "structure": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
            "adapters": {
                "A": "AGCTTACTTGTGAAGA",
                "B": "ACTTGTAAGCTGTCTA",
                "C": "ACTCTGTCAGGTCCGA",
                "D": "ACCTCCTCCTCCAGAA",
                "E": "AACCGGACACACTTAG",
                "F": "AGAGTCCAATTCGCAG",
                "G": "AATCAAGGCTTAACGG",
                "H": "ATGTTGAATCCTAGCG",
                "I": "AGTGCGTTGCGAATTG",
                "J": "AATTGCGTAGTTGGCC",
                "K": "ACACTTGGTCGCAATC",
                "L": "AGTAAGCCTTCGTGTC",
                "M": "ACCTAGATCAGAGCCT",
                "N": "AGGTATGCCGGTTAAG",
                "O": "AAGTCACCGGCACCTT",
                "P": "ATGAAGTGGCTCGAGA",
            },
            "deprecated": False,
            "name": "mas_15",
        },
        "mas_10": {
            "description": "10-element MAS-ISO-seq array",
            "version": "3.0.0",
            "structure": ["Q", "C", "M", "I", "O", "J", "B", "D", "K", "H", "R"],
            "adapters": {
                "Q": "AAGCACCATAATGTGT",
                "C": "ACTCTGTCAGGTCCGA",
                "M": "ACCTAGATCAGAGCCT",
                "I": "AGTGCGTTGCGAATTG",
                "O": "AAGTCACCGGCACCTT",
                "J": "AATTGCGTAGTTGGCC",
                "B": "ACTTGTAAGCTGTCTA",
                "D": "ACCTCCTCCTCCAGAA",
                "K": "ACACTTGGTCGCAATC",
                "H": "ATGTTGAATCCTAGCG",
                "R": "AACCGGACACACTTAG",
            },
            "deprecated": False,
            "name": "mas_10",
        },
        "isoseq": {
            "description": "PacBio IsoSeq model",
            "version": "3.0.0",
            "structure": ["V", "M"],
            "adapters": {
                "V": "TCTACACGACGCTCTTCCGATCT",
                "M": "GTACTCTGCGTTGATACCACTGCTT",
            },
            "deprecated": False,
            "name": "isoseq",
        },
    },
    'cdna': {
        "sc_10x3p": {
            "description": "single-cell 10x 3' kit",
            "version": "3.0.0",
            "structure": ["5p_Adapter", "CBC", "UMI", "Poly_T", "cDNA", "3p_Adapter"],
            "adapters": {
                "5p_Adapter": "TCTACACGACGCTCTTCCGATCT",
                "CBC": {FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME: 16},
                "UMI": {FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME: 12},
                "Poly_T": {HPR_SEGMENT_TYPE_NAME: ("T", 30)},
                "cDNA": RANDOM_SEGMENT_NAME,
                "3p_Adapter": "CCCATGTACTCTGCGTTGATACCACTGCTT",
            },
            "named_random_segments": ["CBC", "UMI", "cDNA"],
            "coding_region": "cDNA",
            "annotation_segments": {
                "UMI": [(longbow.utils.constants.READ_UMI_TAG, longbow.utils.constants.READ_UMI_POS_TAG),
                        (longbow.utils.constants.READ_RAW_UMI_TAG, longbow.utils.constants.READ_UMI_POS_TAG)],
                "CBC": [(longbow.utils.constants.READ_BARCODE_TAG,
                         longbow.utils.constants.READ_BARCODE_POS_TAG),
                        (longbow.utils.constants.READ_RAW_BARCODE_TAG,
                         longbow.utils.constants.READ_BARCODE_POS_TAG)],
            },
            "deprecated": False,
            "name": "sc_10x3p",
        },
        "sc_10x5p": {
            "description": "single-cell 10x 5' kit",
            "version": "3.0.0",
            "structure": ["5p_Adapter", "CBC", "UMI", "SLS", "cDNA", "Poly_A", "3p_Adapter"],
            "adapters": {
                "5p_Adapter": "TCTACACGACGCTCTTCCGATCT",
                "CBC": {FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME: 16},
                "UMI": {FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME: 10},
                "SLS": "TTTCTTATATGGG",
                "cDNA": RANDOM_SEGMENT_NAME,
                "Poly_A": {HPR_SEGMENT_TYPE_NAME: ("A", 30)},
                "3p_Adapter": "GTACTCTGCGTTGATACCACTGCTT",
            },
            "named_random_segments": ["CBC", "UMI", "cDNA"],
            "coding_region": "cDNA",
            "annotation_segments": {
                "UMI": [(longbow.utils.constants.READ_UMI_TAG, longbow.utils.constants.READ_UMI_POS_TAG),
                        (longbow.utils.constants.READ_RAW_UMI_TAG, longbow.utils.constants.READ_UMI_POS_TAG)],
                "CBC": [(longbow.utils.constants.READ_BARCODE_TAG, longbow.utils.constants.READ_BARCODE_POS_TAG),
                        (longbow.utils.constants.READ_RAW_BARCODE_TAG,
                         longbow.utils.constants.READ_BARCODE_POS_TAG)],
            },
            "deprecated": False,
            "name": "sc_10x5p",
        },
        "bulk_10x5p": {
            "description": "bulk 10x 5' kit",
            "version": "3.0.0",
            "structure": ["5p_Adapter", "UMI", "SLS", "cDNA", "Poly_A", "sample_index", "3p_Adapter"],
            "adapters": {
                "5p_Adapter": "TCTACACGACGCTCTTCCGATCT",
                "UMI": {FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME: 10},
                "SLS": "TTTCTTATATGGG",
                "cDNA": RANDOM_SEGMENT_NAME,
                "Poly_A": {HPR_SEGMENT_TYPE_NAME: ("A", 30)},
                "sample_index": {FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME: 10},
                "3p_Adapter": "CTCTGCGTTGATACCACTGCTT",
            },
            "named_random_segments": ["UMI", "cDNA", "sample_index"],
            "coding_region": "cDNA",
            "annotation_segments": {
                "UMI": [(longbow.utils.constants.READ_UMI_TAG, longbow.utils.constants.READ_UMI_POS_TAG),
                        (longbow.utils.constants.READ_RAW_UMI_TAG, longbow.utils.constants.READ_UMI_POS_TAG)],
                "sample_index": [(longbow.utils.constants.READ_DEMUX_TAG,
                                  longbow.utils.constants.READ_DEMUX_POS_TAG)],
            },
            "deprecated": False,
            "name": "bulk_10x5p",
        },
        "bulk_teloprimeV2": {
            "description": "Lexogen TeloPrime V2 kit",
            "version": "3.0.0",
            "structure": ["TPV2_adapter", "cDNA", "Poly_A", "idx", "rev_bind"],
            "adapters": {
                "TPV2_adapter": "CTACACGACGCTCTTCCGATCTTGGATTGATATGTAATACGACTCACTATAG",
                "cDNA": RANDOM_SEGMENT_NAME,
                "Poly_A": {HPR_SEGMENT_TYPE_NAME: ("A", 30)},
                "idx": {FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME: 10},
                "rev_bind": "CTCTGCGTTGATACCACTGCTT",
            },
            "named_random_segments": ["idx", "cDNA"],
            "coding_region": "cDNA",
            "annotation_segments": {
                "idx": [(longbow.utils.constants.READ_INDEX_TAG, longbow.utils.constants.READ_BARCODE_POS_TAG)],
            },
            "deprecated": False,
            "name": "bulk_teloprimeV2",
        },
        "spatial_slideseq": {
            "description": "Slide-seq protocol",
            "version": "3.0.0",
            "structure": ["5p_Adapter", "SBC2", "SLS2", "SBC1", "UMI", "Poly_T", "cDNA", "3p_Adapter"],
            "adapters": {
                "5p_Adapter": "TCTACACGACGCTCTTCCGATCT",
                "SBC2": {FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME: 8},
                "SLS2": "TCTTCAGCGTTCCCGAGA",
                "SBC1": {FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME: 6},
                "UMI": {FIXED_LENGTH_RANDOM_SEGMENT_TYPE_NAME: 9},
                "Poly_T": {HPR_SEGMENT_TYPE_NAME: ("T", 30)},
                "cDNA": RANDOM_SEGMENT_NAME,
                "3p_Adapter": "CCCATGTACTCTGCGTTGATACCACTGCTT",
            },
            "named_random_segments": ["UMI", "SBC2", "SBC1", "cDNA"],
            "coding_region": "cDNA",
            "annotation_segments": {
                "UMI": [(longbow.utils.constants.READ_UMI_TAG, longbow.utils.constants.READ_UMI_POS_TAG),
                        (longbow.utils.constants.READ_RAW_UMI_TAG, longbow.utils.constants.READ_UMI_POS_TAG)],
                "SBC1": [(longbow.utils.constants.READ_SPATIAL_BARCODE1_TAG,
                         longbow.utils.constants.READ_SPATIAL_BARCODE1_POS_TAG)],
                "SBC2": [(longbow.utils.constants.READ_SPATIAL_BARCODE2_TAG,
                         longbow.utils.constants.READ_SPATIAL_BARCODE2_POS_TAG)],
            },
            "deprecated": False,
            "name": "spatial_slideseq",
        },
    }
}
