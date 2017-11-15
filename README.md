# cf-vae
This is a re-implemenmtation of
Xiaopeng Li and James She. Collaborative Variational Autoencoder for Recommder Systems.
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2017 (KDD'17).

# Prerequisites:
tensorflow
tensorbayes : pip install tensorbayes

Change the initializer of "tensorbayes.layers.dense" to "xavier_initializer" as the default initializer sometimes doesn't works well

Add function: binary_crossentropy to tensorbayes.tfutils, we will use this function
def binary_crossentropy(prob, labels, eps=1e-9):
    return -(labels * tf.log(prob + eps) + (1.0 - labels) * tf.log(1.0 - prob + eps))


# Notes
This implementation focuses on simplicity and ease of understanding, may be less efficient. Codes are written similiar to the formulars given in the paper in a EM style.

There are 4 main differences:
1) We didn't use layer by layer pre-training of vae, nor we use a symmetric scheme(use transpose of weights in inference net as weights of generative network)
2) We didn't add mask noise to the inputs and train in a SDAE style
3) We didn't do weight decay for vae weights parameters, we think kl penalty with give enough regularization
3) We run less epochs, just 3 EM epochs

We found that 1), 2) are important for good performance, without these training technics the performence dropped around 3 points.
Nevertheless, it perform still better than all its competitives


# Test
First run "train_vae.py", this will give a pretrained model.
Run "train_cvae.py", this will load this pre-trained model and then fine tune for cf-vae
