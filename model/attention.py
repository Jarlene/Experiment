from argparse import Namespace
from turtle import forward

import torch
from torch import nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super(self, Attention).__init__()
        self.args = args
        self.context = args.context
        self.atten_type = args.atten_type
        self.alignment = args.alignment

        if self.context not in ['many-to-many', 'many-to-one']:
            raise ValueError("Argument for param @context is not recognized")

        if self.atten_type not in ['global', 'local-m', 'local-p', 'local-p*', 'self']:
            raise ValueError(
                "Argument for param @alignment_type is not recognized")

        if self.atten_type == 'global' and args.window_width is not None:
            raise ValueError(
                "Can't use windowed approach with global attention")
        if self.context == 'many-to-many' and self.atten_type == 'local-p*':
            raise ValueError(
                "Can't use local-p* approach in many-to-many scenarios")
        if self.alignment not in ['dot', 'general', 'location', 'concat', 'scaled_dot']:
            raise ValueError(
                "Argument for param @score_function is not recognized")

        self.input_sequence_length, self.hidden_dim = args.input_sequence_length, args.hidden_dim

        if self.atten_type == 'local-p':
            self.W_p = Linear(self.hidden_dim, self.hidden_dim)
            self.v_p = Linear(self.hidden_dim, 1)

        if 'dot' not in self.alignment:
            self.W_a = Linear(self.hidden_dim, self.hidden_dim)

        if self.alignment == 'concat':
            self.U_a = Linear(self.hidden_dim, self.hidden_dim)
            self.v_a = Linear(self.hidden_dim, 1)

    def forward(self, inputs):
        if self.context == 'many-to-one':
            # (B, H)
            target_hidden_state = inputs[1]
            # (B, S, H)
            source_hidden_states = inputs[0]
        elif self.context == 'many-to-many':
            # (B, H)
            target_hidden_state = inputs[1]
            # (B, S, H)
            source_hidden_states = inputs[0]

            current_timestep = inputs[2]

        target_hidden_state = target_hidden_state.unsqueeze(1)

        if self.atten_type == 'global':
            # (B, S, H)
            source_hidden_states = source_hidden_states

        elif 'local' in self.atten_type:
            self.window_width = 8 if self.args.window_width is None else self.args.window_width

            if self.atten_type == 'local-m':
                if self.context == 'many-to-one':
                    aligned_position = self.input_sequence_length
                elif self.context == 'many-to-many':
                    aligned_position = current_timestep
                # Get window borders
                left = int(aligned_position - self.window_width
                           if aligned_position - self.window_width >= 0
                           else 0)
                right = int(aligned_position + self.window_width
                            if aligned_position + self.window_width <= self.input_sequence_length
                            else self.input_sequence_length)
                # Extract window window
                def extract_func(x): return x[:, left:right, :]
                source_hidden_states = extract_func(
                    source_hidden_states)  # (B, S*=(D, 2xD), H)
            elif self.atten_type == 'local-p':
                aligned_position = self.W_p(target_hidden_state)
                aligned_position = torch.tanh(aligned_position)  # (B, 1, H)
                # (B, 1, 1)
                aligned_position = self.v_p(aligned_position)
                aligned_position = torch.sigmoid(aligned_position)  # (B, 1, 1)
                aligned_position = aligned_position * self.input_sequence_length

            elif self.alignment_type == 'local-p*':
                # (B, S, H)
                aligned_position = self.W_p(source_hidden_states)
                aligned_position = torch.tanh(aligned_position)  # (B, S, H)
                # (B, S, 1)
                aligned_position = self.v_p(aligned_position)
                aligned_position = torch.sigmoid(aligned_position)  # (B, S, 1)
                # Only keep top D values out of the sigmoid activation, and zero-out the rest
                aligned_position = torch.squeeze(
                    aligned_position, dim=-1)  # (B, S)
                top_probabilities = torch.topk(aligned_position,  # (values:(B, D), indices:(B, D))
                                               k=self.window_width,
                                               sorted=False)
                onehot_vector = F.one_hot(top_probabilities.indices,
                                          self.input_sequence_length)       # (B, D, S)
                # (B, S)
                onehot_vector = torch.sum(onehot_vector, dim=1)
                aligned_position = torch.multiply(
                    aligned_position, onehot_vector)  # (B, S)
                aligned_position = torch.unsqueeze(
                    aligned_position, dim=-1)                        # (B, S, 1)
                # (B, S, 1)
                initial_source_hidden_states = source_hidden_states
                source_hidden_states = torch.multiply(
                    source_hidden_states, aligned_position)         # (B, S*=S(D), H)
                # Scale back-to approximately original hidden state values
                # (B, S, 1)
                aligned_position += torch.epsilon()
                # (B, S*=S(D), H)
                source_hidden_states /= aligned_position
                source_hidden_states = initial_source_hidden_states + \
                    source_hidden_states          # (B, S, H)

            if 'dot' in self.alignment:
                attention_score = source_hidden_states * \
                    target_hidden_state        # (B, S*, 1)
            if self.alignment == 'scaled_dot':
                attention_score *= 1 / np.sqrt(
                    float(source_hidden_states.shape[2]))     # (B, S*, 1)
            elif self.score_function == 'general':
                weighted_hidden_states = self.W_a(
                    source_hidden_states)                                 # (B, S*, H)
                attention_score = weighted_hidden_states * \
                    target_hidden_state     # (B, S*, 1)
