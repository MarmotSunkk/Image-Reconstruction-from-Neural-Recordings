This script can train a Stable Diffusion model to reconstruct the visual stimulus images received by mice through neural recording. The model is saved in the allen directory, where ddpm_ckpt_neuron_cond.pth is a Stable Diffusion model that reconstructs the image. The generated condition is a 25×1 neural vector, which is the result of dimensionality reduction using CEBRA. vqvae_autoencoder_ckpt.pth is a VQ-VAE model that compresses the original image into latent space.

When using this script, first you want to use tools/train_vqvae.py to train a VQ-VAE model. Then you should use tools/train_ddpm_cond.py to train a stable diffusion model based on conditional generation. Finally, you can run sample_ddpm_text_cond.py to generate the image. Note that you need to specify the generated condition in this file.

When training and using the model, you can edit the corresponding yaml file in the config folder to freely change your model configuration, which is necessary when using different datasets.

When training the VQ-VAE model, I used about 10,000 randomly selected images from the iNatualist dataset.

When training the Stable Diffusion model, the visual stimulus images from session 791319847 in the Allen Visual Coding database were used. The images are named in the order of visual stimuli. Note that pure black images and corresponding neural activities are removed. The neural responses should be stored in a csv file with the index corresponding to the image name, and each row has a column named 'frame', which is the index information of the image.

Due to its large size, the dataset and the trained model are not provided in this resource.
