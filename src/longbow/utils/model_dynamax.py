"""Dynamax-based Hidden Markov Model implementation for Longbow.

This module provides a LibraryModel class using dynamax's CategoricalHMM
for sequence annotation and decoding tasks.
"""

import logging
import re
import json
import sys

import jax
import jax.numpy as jnp
import jax.random as jr

from dynamax.hidden_markov_model import CategoricalHMM
from dynamax.hidden_markov_model.inference import hmm_posterior_mode

# Import model_utils_dynamax for pre-configured models
from .model_utils_dynamax import ModelBuilder

logging.basicConfig(stream=__import__('sys').stderr)
logger = logging.getLogger(__name__)

# Model name delimiter (same as pomegranate version)
MODEL_DESC_DELIMITER = "+"

# Regex for parsing model names (same as pomegranate version)
MODEL_NAME_REGEX = re.compile(f"\\{MODEL_DESC_DELIMITER}")


class LibraryModel:
    """Model describing a library preparation using dynamax CategoricalHMM.

    The model can annotate the known sections of a read from the library it describes.
    """

    def __init__(self, n_states, n_emissions, state_names=None):
        """Initialize the HMM model.

        Args:
            n_states: Number of hidden states in the model
            n_emissions: Number of emission categories (e.g., 4 for nucleotides)
            state_names: Optional list of state names. If None, states are named
                         as "State_0", "State_1", ..., "State_{n_states-1}"
        """
        self.n_states = n_states
        self.n_emissions = n_emissions

        # Initialize state name mappings
        if state_names is not None:
            self.state_names = {name: idx for idx, name in enumerate(state_names)}
            self.states = list(state_names)
        else:
            self.state_names = {f"State_{i}": i for i in range(n_states)}
            self.states = [f"State_{i}" for i in range(n_states)]

        # Placeholder for the dynamax model
        self.model = None
        self.params = None

    def build(self, states, initial_probs, trans_matrix, emissions_matrix):
        """Build the HMM model with the given parameters.

        Args:
            states: List of state names (must match self.states)
            initial_probs: Initial state probability distribution, shape (n_states,)
            trans_matrix: Transition matrix, shape (n_states, n_states)
            emissions_matrix: Emission probability matrix, shape (n_states, n_emissions)
        """
        # Validate inputs
        if len(states) != self.n_states:
            raise ValueError(f"Number of states mismatch: expected {self.n_states}, got {len(states)}")

        if initial_probs.shape != (self.n_states,):
            raise ValueError(f"Initial probabilities shape mismatch: expected ({self.n_states},), "
                           f"got {initial_probs.shape}")

        if trans_matrix.shape != (self.n_states, self.n_states):
            raise ValueError(f"Transition matrix shape mismatch: expected ({self.n_states}, {self.n_states}), "
                           f"got {trans_matrix.shape}")

        if emissions_matrix.shape != (self.n_states, self.n_emissions):
            raise ValueError(f"Emission matrix shape mismatch: expected ({self.n_states}, {self.n_emissions}), "
                           f"got {emissions_matrix.shape}")

        # Update state name mappings if provided
        if states is not None and len(states) == self.n_states:
            self.state_names = {name: idx for idx, name in enumerate(states)}
            self.states = list(states)

        # Create the dynamax CategoricalHMM
        # emission_dim=1 for univariate emissions (like nucleotides)
        self.model = CategoricalHMM(
            num_states=self.n_states,
            emission_dim=1,
            num_classes=self.n_emissions
        )

        # Initialize the model with our parameters
        # Reshape emissions_matrix to (n_states, 1, n_emissions) for dynamax
        emissions_probs = jnp.array(emissions_matrix).reshape(self.n_states, 1, self.n_emissions)
        initial_probs_arr = jnp.array(initial_probs)
        trans_matrix_arr = jnp.array(trans_matrix)

        self.params, self.props = self.model.initialize(
            key=jr.PRNGKey(42),
            initial_probs=initial_probs_arr,
            transition_matrix=trans_matrix_arr,
            emission_probs=emissions_probs
        )

    def annotate(self, seq):
        """Annotate the given sequence using Viterbi decoding.
        
        This method returns the path in a format compatible with the original pomegranate version,
        where consecutive states with the same operation are merged into CIGAR format.

        Args:
            seq: Sequence of emissions (integers representing categories).
                 For DNA/RNA, typically: A=0, C=1, G=2, T=3 (or similar mapping)

        Returns:
            tuple: (log_prob, list of path segments in CIGAR format)
                   Path segments are like ['5p_Adapter:D1M22', 'CBC:M16', ...]
        """
        if self.model is None or self.params is None:
            raise RuntimeError("Model has not been built. Call build() first.")

        # Convert sequence to JAX array and ensure correct shape
        if isinstance(seq, (list, tuple)):
            seq = jnp.array(seq)

        # Ensure sequence is 1D integer array
        if seq.ndim == 2:
            seq = seq.squeeze()
        if seq.ndim != 1:
            raise ValueError(f"Sequence must be 1D, got shape {seq.shape}")

        # Get the most likely state sequence using Viterbi (hmm_posterior_mode)
        most_likely_states = self.model.most_likely_states(self.params, seq)

        # Convert state indices to state names
        state_path = [self.states[state_idx] for state_idx in most_likely_states]

        # Compute log probability
        log_prob = float(self.model.marginal_log_prob(self.params, seq))

        # Convert the per-position state path to cumulative CIGAR format
        # This mimics the behavior of pomegranate's annotate method
        ppath = self._state_path_to_cigar(state_path)

        return log_prob, ppath

    def _state_path_to_cigar(self, state_path):
        """Convert a per-position state path to CIGAR format.
        
        This method converts states like:
            ['5p_Adapter:I0', '5p_Adapter:D1', '5p_Adapter:D2', '5p_Adapter:I3', ...]
        Into CIGAR format like:
            ['5p_Adapter:I1D2I1', 'CBC:M16', ...]
        
        Args:
            state_path: List of state names from Viterbi decoding
            
        Returns:
            list: List of segments in CIGAR format
        """
        import re
        
        ppath = []
        cigar = []
        cur_adapter_name = ''
        cur_op = ''
        cur_op_len = 0
        
        for state in state_path:
            # Skip start/end indicators if present
            if '-start' in state or '-end' in state:
                continue
                
            # Parse state name: "AdapterName:OpIndex" -> ("AdapterName", "Op", index)
            # e.g., "5p_Adapter:I0" -> ("5p_Adapter", "I", 0)
            # e.g., "5p_Adapter:D1" -> ("5p_Adapter", "D", 1)
            match = re.match(r'([^:]+):([A-Z]+)(\d+)', state)
            if match:
                adapter_name = match.group(1)
                op = match.group(2)
                idx = int(match.group(3))
            else:
                # Try alternate format (e.g., "random:RDA")
                if ':' in state:
                    parts = state.split(':', 1)
                    adapter_name = parts[0]
                    op = parts[1] if len(parts) > 1 else ''
                else:
                    adapter_name = state
                    op = ''
                idx = -1
            
            # Track adapter changes
            if adapter_name != cur_adapter_name:
                if cur_adapter_name != '':
                    cigar.append(f'{cur_op}{cur_op_len}')
                    ppath.append(f'{cur_adapter_name}:{"".join(cigar)}')
                
                cigar = []
                cur_adapter_name = adapter_name
                cur_op = op
                cur_op_len = 0
            
            # Track operation changes within same adapter
            if op != cur_op:
                if cur_op != '':
                    cigar.append(f'{cur_op}{cur_op_len}')
                cur_op = op
                cur_op_len = 0
            
            cur_op_len += 1
        
        # Add the last segment
        if cur_adapter_name != '':
            cigar.append(f'{cur_op}{cur_op_len}')
            ppath.append(f'{cur_adapter_name}:{"".join(cigar)}')
        
        return ppath

    def annotate_with_log_prob(self, seq):
        """Annotate the given sequence and return states with log probability.

        Args:
            seq: Sequence of emissions (integers representing categories)

        Returns:
            tuple: (log_prob, list of state names)
        """
        if self.model is None or self.params is None:
            raise RuntimeError("Model has not been built. Call build() first.")

        # Convert sequence to JAX array
        if isinstance(seq, (list, tuple)):
            seq = jnp.array(seq)

        if seq.ndim == 2:
            seq = seq.squeeze()
        if seq.ndim != 1:
            raise ValueError(f"Sequence must be 1D, got shape {seq.shape}")

        # Get the most likely state sequence
        most_likely_states = self.model.most_likely_states(self.params, seq)

        # Compute log probability
        log_prob = self.model.marginal_log_prob(self.params, seq)

        # Convert state indices to state names
        state_path = [self.states[state_idx] for state_idx in most_likely_states]

        return float(log_prob), state_path

    def dump_as_dotfile(self, filename, do_subgraphs=True):
        """Export the model as a DOT format visualization file.

        Args:
            filename: Output filename for the DOT file
            do_subgraphs: If True, create subgraphs for each state group
        """
        with open(filename, 'w') as f:
            f.write(f"digraph HMMModel {{\n")
            f.write("    rankdir=LR;\n")
            f.write("    node [shape=box];\n\n")

            # Get parameters as numpy arrays
            initial_probs = jnp.array(self.params.initial.probs)
            trans_matrix = jnp.array(self.params.transitions.transition_matrix)
            emission_probs = jnp.array(self.params.emissions.probs)

            # Write out states as nodes
            f.write("    // States:\n")
            for i, state_name in enumerate(self.states):
                # Include initial probability in label
                init_p = initial_probs[i]
                f.write(f'    "{state_name}" [label="{state_name}\\n(init={init_p:.3f})"];\n')
            f.write("\n")

            # Write out transitions as edges
            f.write("    // Transitions:\n")
            for i in range(self.n_states):
                for j in range(self.n_states):
                    prob = float(trans_matrix[i, j])
                    if prob > 0.001:  # Only show significant transitions
                        f.write(f'    "{self.states[i]}" -> "{self.states[j]}" '
                               f'[label="{prob:.3f}", weight={prob}];\n')
            f.write("\n")

            # Write emission information as a separate section
            f.write("    // Emissions (shown as tooltip):\n")
            for i, state_name in enumerate(self.states):
                probs = emission_probs[i, 0, :] if emission_probs.ndim == 3 else emission_probs[i]
                probs_str = ", ".join([f"{p:.3f}" for p in probs])
                f.write(f'    "{state_name}" [tooltip="{state_name}: [{probs_str}]"];\n')

            f.write("}\n")

        logger.info(f"Model exported to {filename}")

    def dump_as_dotfile_simple(self, filename=None):
        """Export the model as a simple DOT format file.

        Args:
            filename: Output filename. If None, generates a default name.
        """
        if filename is None:
            filename = "hmm_model.dot"
        self.dump_as_dotfile(filename, do_subgraphs=False)

    def to_dict(self):
        """Serialize the model to a dictionary.

        Returns:
            dict: Model parameters and metadata
        """
        if self.params is None:
            return {
                "n_states": self.n_states,
                "n_emissions": self.n_emissions,
                "state_names": self.states,
                "model_built": False
            }

        initial_probs = jnp.array(self.params.initial.probs)
        trans_matrix = jnp.array(self.params.transitions.transition_matrix)
        emission_probs = jnp.array(self.params.emissions.probs)

        # Squeeze emission_probs if needed
        if emission_probs.ndim == 3:
            emission_probs = emission_probs.squeeze(1)

        return {
            "n_states": self.n_states,
            "n_emissions": self.n_emissions,
            "state_names": self.states,
            "initial_probs": initial_probs.tolist(),
            "trans_matrix": trans_matrix.tolist(),
            "emissions_matrix": emission_probs.tolist(),
            "model_built": True
        }

    def get_state_index(self, state_name):
        """Get the index for a given state name.

        Args:
            state_name: Name of the state

        Returns:
            int: Index of the state

        Raises:
            KeyError: If state_name is not found
        """
        return self.state_names[state_name]

    def get_state_name(self, index):
        """Get the name for a given state index.

        Args:
            index: Index of the state

        Returns:
            str: Name of the state

        Raises:
            IndexError: If index is out of range
        """
        return self.states[index]

    @property
    def is_built(self):
        """Check if the model has been built.

        Returns:
            bool: True if model is built, False otherwise
        """
        return self.model is not None and self.params is not None

    # =========================================================================
    # Methods for compatibility with pomegranate version (bam_utils.load_model)
    # =========================================================================

    @staticmethod
    def has_prebuilt_model(model_name):
        """Check if the given model name is a pre-configured model.

        Args:
            model_name: Model name (e.g., 'mas_15+sc_10x5p')

        Returns:
            bool: True if model is pre-configured, False otherwise
        """
        model_name_pieces = model_name.split(MODEL_DESC_DELIMITER)

        if len(model_name_pieces) == 2:
            array_model_name, cdna_model_name = model_name_pieces

            if array_model_name not in ModelBuilder.pre_configured_models['array'].keys():
                return False

            if cdna_model_name not in ModelBuilder.pre_configured_models['cdna'].keys():
                return False

            return True

        return False

    @staticmethod
    def build_pre_configured_model(model_name):
        """Build a pre-configured model by name.

        Args:
            model_name: Model name (e.g., 'mas_15+sc_10x5p')

        Returns:
            LibraryModel: A fully built model
        """
        (array_model_name, cdna_model_name) = model_name.split(MODEL_DESC_DELIMITER, 2)

        return LibraryModel.from_json_obj({
            'array': ModelBuilder.pre_configured_models['array'][array_model_name],
            'cdna': ModelBuilder.pre_configured_models['cdna'][cdna_model_name],
            'name': model_name,
            'description': f"{ModelBuilder.pre_configured_models['array'][array_model_name]['description']}, "
                          f"{ModelBuilder.pre_configured_models['cdna'][cdna_model_name]['description']}"
        })

    @staticmethod
    def from_json_file(json_file):
        """Create a LibraryModel instance from the given json file.

        Args:
            json_file: Path to the JSON file

        Returns:
            LibraryModel: A fully built model
        """
        try:
            with open(json_file) as f:
                json_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"File does not exist: {json_file}")
            sys.exit(1)

        return LibraryModel.from_json_obj(json_data)

    @staticmethod
    def from_json_obj(json_data):
        """Create a LibraryModel instance from the given JSON data.

        Args:
            json_data: JSON object with 'array', 'cdna', 'name', 'description'

        Returns:
            LibraryModel: A fully built model
        """
        # Validate required fields
        required_fields = ["array", "cdna", "name", "description"]
        missing_fields = [f for f in required_fields if f not in json_data]
        if missing_fields:
            message = f"ERROR: JSON model is missing required fields: {','.join(missing_fields)}"
            logger.critical(message)
            raise RuntimeError(message)

        # Build the model using ModelBuilder.make_full_longbow_model
        model = ModelBuilder.make_full_longbow_model(
            array_model=json_data['array'],
            cdna_model=json_data['cdna'],
            model_name=json_data['name']
        )

        # Set the description
        model.description = json_data['description']
        model.name = json_data['name']

        # Store adapter dict for compatibility
        model.adapter_dict = {**json_data['array']['adapters'], **json_data['cdna']['adapters']}
        model.array_model = json_data['array']
        model.cdna_model = json_data['cdna']
        model.named_random_segments = set(json_data['cdna'].get('named_random_segments', []))
        model.coding_region = json_data['cdna'].get('coding_region')
        model.annotation_segments = json_data['cdna'].get('annotation_segments', {})

        return model

    def to_json(self, indent=4):
        """Serialize this model to JSON.

        Args:
            indent: JSON indentation

        Returns:
            str: JSON string representation
        """
        if hasattr(self, 'array_model') and hasattr(self, 'cdna_model'):
            model_data = {
                "name": self.name,
                "description": self.description,
                "array": self.array_model,
                "cdna": self.cdna_model
            }
            return json.dumps(model_data, indent=indent)
        else:
            # Fall back to basic serialization
            return json.dumps(self.to_dict(), indent=indent)
