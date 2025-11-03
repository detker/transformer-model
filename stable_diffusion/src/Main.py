from VAE import VAEEncoder, VAEDecoder
from diffusion import Diffusion
from clip import CLIP
import model_converter

class Main():
    def preload_models_from_std_weights(self, path, device):
        state_dict = model_converter.load_from_std_weights(path, device)

        encoder = VAEEncoder().to(device)
        # print('ENCODER DICT:', encoder.state_dict().keys())
        encoder.load_state_dict(state_dict['encoder'], strict=True)

        decoder = VAEDecoder().to(device)
        decoder.load_state_dict(state_dict['decoder'], strict=True)

        diffusion = Diffusion().to(device)
        diffusion.load_state_dict(state_dict['diffusion'], strict=True)

        clip = CLIP().to(device)
        clip.load_state_dict(state_dict['clip'], strict=True)

        return {'encoder': encoder,
                'decoder': decoder,
                'diffusion': diffusion,
                'clip': clip}



