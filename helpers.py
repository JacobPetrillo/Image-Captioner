import spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import random, os, string
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from torchtext.vocab import GloVe
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np
import torch.nn as nn
import json



## Vocabulary, MyCollate, save_checkpoint, and load_checkpoint are from
## https://youtu.be/y2BaTt1fxJU?si=ipC6wbwDsbIxWMmR
## I kept them as they were cleaner than my implementation of achieving these functions

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1,"<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        text = str(text)
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)if tok.text.isalpha() and tok.text not in string.punctuation]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4 #start after the 4 special tokens

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return[self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
               for token in tokenized_text]
    
    def save_vocab(self, file_path):
        data = {
            'itos': self.itos,
            'stoi': self.stoi,
            'freq_threshold': self.freq_threshold
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_vocab(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        vocab = cls(data['freq_threshold'])
        vocab.itos = {int(k): v for k, v in data['itos'].items()}
        vocab.stoi = data['stoi']
        return vocab


  
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets    
    


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)



def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step



# For the option of using pre-trained word embeddings (could replace GloVe with different ones)
def load_pretrained_embeddings(vocab, glove_dim=300):
    # Load GloVe embeddings
    glove = GloVe(name="840B", dim=glove_dim)
    
    # Initialize embeddings tensor (default random initialization)
    embeddings = torch.randn(len(vocab), glove_dim)
    
    for i, word in vocab.itos.items():
        if word in glove.stoi:
            embeddings[i] = glove[word]
            
    return embeddings


# To help with viewing the image
def tensor_to_pil_with_denormalization(tensor):
    """Convert tensor to denormalized PIL image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device)
    denormalized = tensor * std[:, None, None] + mean[:, None, None]
    denormalized = torch.clamp(denormalized, 0, 1)
    pil_img = ToPILImage()(denormalized)
    return pil_img


# Shows the original image, and the preprocessed image in two different forms, to help show how the computer might "see" the images
def visualize_dataset(dataset, vocabulary, num_images_to_display=3):

    num_images_to_display = num_images_to_display
    rand_indices = random.sample(range(len(dataset)), num_images_to_display)

    to_pil = ToPILImage()

    for index in rand_indices:
        # Create a subplot with three columns (original, preprocessed, and denormalized preprocessed)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        
        # Get the image and its associated captions
        img, target = dataset[index]

        # Convert the tensor to a PIL image for the denormalized (unprocessed) image
        img_pil = tensor_to_pil_with_denormalization(img)
        axes[0].imshow(img_pil)
        axes[0].axis('off')
        axes[0].set_title(f"Image {index + 1} (Original)")
        
        # Convert the tensor to a PIL image for the preprocessed image
        axes[1].imshow(torch.clamp(img.permute(1, 2, 0), 0, 1))
        axes[1].axis('off')
        axes[1].set_title(f"Image {index + 1} Preprocessed (Tensor)")
        
        # Display the preprocessed image directly from tensor (rainbow effect)
        img_pil_preprocessed = to_pil(img)
        axes[2].imshow(img_pil_preprocessed)
        axes[2].axis('off')
        axes[2].set_title(f"Image {index + 1} Preprocessed (PIL)")

        if target.dim() == 1 or isinstance(target[0], int):
            word_captions = ' '.join([vocabulary.itos[token.item()] for token in target])
        else:
            word_captions = [' '.join([vocabulary.itos[token.item()] for token in caption]) for caption in target]
            word_captions = '\n'.join(word_captions)


        # Display the captions below the images (centered between them)
        fig.text(0.5, 0.01, word_captions, ha="center", va="bottom", size=15)
        
        plt.tight_layout(pad=3.0)
        plt.show()
        plt.close(fig)


# Shows the target caption and a caption generated by the model for a given image
def visualize_caption_generation(model, data_loader, vocab, max_len, device, numDisplayed=0):
    for _, (imgs, captions) in enumerate(data_loader):
        imgs, captions = imgs.to(device), captions.to(device)
  
        if numDisplayed == 0:
            size = len(imgs)
        else:
            size = numDisplayed
            indices = random.sample(range(len(imgs)), numDisplayed)

        for i in range(0, size):

            if numDisplayed == 0:
                index = i
            else:
                index = indices[i]

            image_to_log = imgs[index].unsqueeze(0).to(device)
            target_caption = [vocab.itos[idx] for idx in captions[:, index].tolist()if idx != vocab.stoi["<pad>"]]
            target_caption = ' '.join(target_caption[1:-1])




            generated_caption = model.caption_image(image_to_log, vocab, max_len)
            caption_string = ' '.join(generated_caption[:-1])  # Convert list of words to a string
            

            # Displaying the image with captions using matplotlib
            img_pil = tensor_to_pil_with_denormalization(image_to_log.squeeze(0))  # Convert the image to a format suitable for plotting
            plt.figure(figsize=(8,8))
            plt.imshow(img_pil)
            plt.title(f"Target: {target_caption}")
            plt.axis('off')  # Hide the axis

            plt.figtext(0.5, 0.0, f"Generated: {caption_string}", ha="center", va="bottom", size=12, wrap=True)
            
            plt.show()

        if numDisplayed != 0:
            break

# Calculates the loss on the validation set
def validation_loss(model, data_loader, criterion, writer, epoch, tfr=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0.0

    with torch.no_grad():
        for imgs, captions in data_loader:
            imgs, captions = imgs.to(device), captions.to(device)
            
            outputs = model(imgs, captions[:-1], tfr)

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[1:, :].reshape(-1))

            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)

    if tfr > 0:
        print(f"Validation Loss after Epoch {epoch}: {avg_loss}")
        writer.add_scalar("Loss/Validation", avg_loss, epoch)
    else:
        print(f"Raw Validation Loss after Epoch {epoch}: {avg_loss}")
        writer.add_scalar("Loss/RawValidation", avg_loss, epoch)
    return avg_loss


# Keeps the logic of logging data to TensorBoard all together.
def log_to_tensorboard(device, model, imgs, captions, vocab, max_len, epoch, batch, writer):
    print(f"Epoch {epoch+1}: Processing batch {batch}")
    # Log a random image and its caption to TensorBoard
    rand_idx = random.randint(0, imgs.size(0) - 1)
    image_to_log = imgs[rand_idx].unsqueeze(0).to(device)

    target_caption =  ' '.join([vocab.itos[idx] for idx in captions[:, rand_idx].tolist() if idx != vocab.stoi["<pad>"]])
    generated_caption = model.caption_image(image_to_log, vocab, max_len)
    caption_string = ' '.join(generated_caption)  # Convert list of words to a string

    writer.add_image(f'Image_at_epoch_{epoch}', image_to_log.squeeze(0), global_step=batch)
    writer.add_text(f'Target/Caption_at_epoch_{epoch}', target_caption, global_step=batch)
    writer.add_text(f'Training/Caption_at_epoch_{epoch}', caption_string, global_step=batch)
            

# Prints and logs to TensorBoard the BLEU 1,2,3 and 4 scores, using corpus_bleu from nltk.translate
def compute_bleu_scores(dataset, model, vocabulary, epoch, max_len=50, writer=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Calculating BLEU Scores...")

    all_references = []
    all_candidates = []
    all_candidates_beam = []

    #dataset.dataset.set_eval_mode()
    dataset.set_eval_mode()

    for idx in range(len(dataset)):
        
        # Get the reference captions
        _, numericalized_captions = dataset[idx]
        references = []
        for cap in numericalized_captions:
            cap = cap.to(device)
            ref = [vocabulary.itos[token.item()] for token in cap]
            ref = ref[1:-1]  # remove <start> and <end> tokens
            references.append(ref)
        
        all_references.append(references)
        
        # Get the generated caption (candidate)
        image, _ = dataset[idx]
        image = image.unsqueeze(0).to(device)
        candidate = model.caption_image(image, vocabulary, max_len)
        candidate = candidate[1:-1]  # remove <start> and <end> tokens
        all_candidates.append(candidate)

    # Compute BLEU scores
    bleu_1 = corpus_bleu(all_references, all_candidates, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    bleu_2 = corpus_bleu(all_references, all_candidates, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1)
    bleu_3 = corpus_bleu(all_references, all_candidates, weights=(0.33, 0.33, 0.33, 0), smoothing_function=SmoothingFunction().method1)
    bleu_4 = corpus_bleu(all_references, all_candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)

    print("BLEU Scores: ")
    print(f"BLEU-1 Score: {bleu_1}")
    print(f"BLEU-2 Score: {bleu_2}")
    print(f"BLEU-3 Score: {bleu_3}")
    print(f"BLEU-4 Score: {bleu_4}")


    # Log to TensorBoard if writer is provided
    if writer and epoch != 0:
        writer.add_scalar("BLEU-1", bleu_1, epoch)
        writer.add_scalar("BLEU-2", bleu_2, epoch)
        writer.add_scalar("BLEU-3", bleu_3, epoch)
        writer.add_scalar("BLEU-4", bleu_4, epoch)

    
    #dataset.dataset.set_train_mode()
    dataset.set_train_mode()
    model.train()

    return bleu_1, bleu_2, bleu_3, bleu_4


# Prints and logs to TensorBoard the METEOR score, using metero_score from nltk.translate
def compute_meteor_score(dataset, model, vocab, epoch, writer=None):
    model.eval()
    #dataset.dataset.set_eval_mode()
    dataset.set_eval_mode()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Calculating METEOR Scores...")
    meteor_scores = []

    # Loop through the dataset
    for img, true_captions in dataset:
        references = [' '.join([vocab.itos[token.item()] for token in cap]).split() for cap in true_captions]

        img = img.unsqueeze(0).to(device)  # Add batch dimension
        predicted_caption = model.caption_image(img, vocab)  # Get the caption prediction from the model
        predicted_caption = predicted_caption[1:-1]
        # Calculate METEOR score
        # Since there can be multiple true captions, we take the max METEOR score
        scores = [meteor_score([ref[1:-1]], predicted_caption) for ref in references]
        meteor_scores.append(max(scores))

    average_meteor = sum(meteor_scores) / len(meteor_scores)

    print(f"METEOR Score: {average_meteor:.4f}")

    # Log to TensorBoard if writer is provided
    if writer and epoch != 0:
        writer.add_scalar("METEOR", average_meteor, epoch)


    #dataset.dataset.set_train_mode()
    dataset.set_train_mode()
    model.train()

    return average_meteor


# Makes a string for the Writer to put the TensorBoard log
def get_logdir(root_dir="runs"):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(root_dir, current_time)
    return logdir


# Used if using scheduled sampling and want to linearly decrease the teacher forcing ratio
def linear_tfr(start_TFR, end_TFR, epoch, total_epochs):
 
    t = epoch / total_epochs
    current_TFR = (1 - t) * start_TFR + t * end_TFR
    return current_TFR
