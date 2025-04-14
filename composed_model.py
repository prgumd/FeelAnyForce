import torch
from torch import nn
from regressor import Regressor
from depth_decoder import Decoder


class ComposedModel(nn.Module):
    """
    Neural network model composed of a tactile backbone, a regressor, and a decoder.
    """
    def __init__(self, args):
        """
        Initializes the ComposedModel with the specified arguments.

        Arguments:
            args: Namespace containing model configuration, such as backbone repo, model name,
                  number of blocks to use, label count, and training options.
        """
        super(ComposedModel, self).__init__()

        self.args = args

        self.tactile_backbone = torch.hub.load(args.tactile_repo, args.tactile_model)
        tactile_embed_dim = self.tactile_backbone.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
            
        self.regressor = Regressor(tactile_embed_dim, num_labels=self.args.num_labels)
            
        self.decoder = Decoder(224, 224, tactile_embed_dim)

        self.regressor.cuda()
        self.tactile_backbone.cuda()
        self.decoder.cuda()


    def get_param_groups(self, lr_backbone, lr_architecture, lr_calibration):
        """
        Returns parameter groups for optimization, potentially assigning different learning rates
        to the backbone, regressor, and decoder depending on training mode.

        Arguments:
            lr_backbone (float): Learning rate for the tactile backbone.
            lr_architecture (float): Learning rate for regressor and decoder.
            lr_calibration (float): Learning rate for calibrating selected layers during fine-tuning.

        Returns:
            List[Dict]: List of parameter groups with their associated learning rates.
        """
        param_groups = []

        if self.args.tactile_backbone_training =='calibration':
            selected_params = []
            num_linear_layers = 0

            # Iterate in reverse to pick the last 'layers_calibration' Linear layers + their LayerNorms
            for layer in reversed(self.regressor.regressor):
                if isinstance(layer, nn.Linear):
                    num_linear_layers += 1
                if num_linear_layers > self.args.layers_calibration:
                    break  

                # Collect trainable parameters (skip GELU but include LayerNorm)
                if isinstance(layer, (nn.Linear, nn.LayerNorm)):
                    for param in layer.parameters():
                        selected_params.append(param)

            param_groups.append({'params': selected_params, 'lr': lr_calibration})
                
        else:
            for _, v in self.tactile_backbone.named_parameters():
                param_groups += [{'params': v, 'lr': lr_backbone}]

            for _, v in self.regressor.named_parameters():
                param_groups += [{'params': v, 'lr': lr_architecture}]
            
            for _, v in self.decoder.named_parameters():
                param_groups += [{'params': v, 'lr': lr_architecture}]

        return param_groups


    def get_encoding(self, model_input):
        """
        Computes the tactile feature encoding from the model input.

        Arguments:
            model_input (Tensor): Input tensor to be passed through the tactile backbone.

        Returns:
            Tensor: Extracted feature representation from the input.
        """
        model = self.tactile_backbone

        if self.args.forward:
            output = model(model_input)
        else:
            intermediate_output = model.get_intermediate_layers(model_input, self.args.n_last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

            if self.args.avgpool_patchtokens:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)

        return output