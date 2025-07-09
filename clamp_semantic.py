import argparse
import subprocess
from clamp_utils import *
from transformers import AutoTokenizer
from unidecode import unidecode

softmax = torch.nn.Softmax(dim=1)


def load_music(filename):
    """
    Load the music from the abc file

    Args:
        filename (str): Path to the abc file

    Returns:
        music (str): Music string
    """

    with open(filename, 'rb') as file:
        file_content = file.read()
    output = file_content.decode('latin-1').replace('\r', '')
    music = unidecode(output).split('\n')
    music = abc_filter(music)

    return music

def abc_filter(lines):
    """
    Filter out the metadata from the abc file

    Args:
        lines (list): List of lines in the abc file
    
    Returns:
        music (str): Music string
    """
    music = ""
    for line in lines:
        if line[:2] in ['A:', 'B:', 'C:', 'D:', 'F:', 'G', 'H:', 'N:', 'O:', 'R:', 'r:', 'S:', 'T:', 'W:', 'w:', 'X:', 'Z:'] \
        or line=='\n' \
        or (line.startswith('%') and not line.startswith('%%score')):
            continue
        else:
            oldline = line
            if "%" in line and not line.startswith('%%score'):
                line = "%".join(line.split('%')[:-1])
                music += line[:-1]
            else:
                music += line
            if oldline != lines[-1] : music += '\n'
    return music

def compute_values( Q_e, K_e, t=1):
    """
    Compute the values for the attention matrix

    Args:
        Q_e (torch.Tensor): Query embeddings
        K_e (torch.Tensor): Key embeddings
        t (float): Temperature for the softmax
    s
    Returns:
        values (torch.Tensor): Values for the attention matrix
    """
    # Normalize the feature representations
    Q_e = torch.nn.functional.normalize(Q_e, dim=1)
    K_e = torch.nn.functional.normalize(K_e, dim=1)

    # Scaled pairwise cosine similarities [1, n]
    logits = torch.mm(Q_e, K_e.T) * torch.exp(torch.tensor(t))
    values = softmax(logits)
    return values.squeeze()


class CLaMP_Semantic():
    
    TEXT_LENGTH = 128
    TEXT_MODEL_NAME = 'distilroberta-base'
    # initialize patchilizer, tokenizer, and softmax
    patchilizer = MusicPatchilizer()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    query_feature = None

    def __init__(self, io_path= "", clamp_model_name = "clamp_sander-wood/clamp-small-512", torch_device="cuda", query=None):
        
        print(f"Requested device: {torch_device}")
        if torch_device.startswith("cuda"):
            if torch.cuda.is_available():    
                print(f"There are {torch.cuda.device_count()} GPU(s) available.")
                self.device = torch.device(torch_device)
            else:
                print(f"No GPU available, using the CPU instead.")
                self.device = torch.device("cpu")
        elif torch_device.startswith("cpu"):
            print("Running in CPU-only mode.")
            self.device = torch.device(torch_device)
        else:
            print("WARNING: Unexpected device name (expected 'cuda' or 'cpu')")
            self.device = torch.device(torch_device)

        
        self.CLAMP_MODEL_NAME = clamp_model_name
        self.QUERY_MODAL = "text"
        self.KEY_MODAL = "music"

        # load CLaMP model
        self.model = CLaMP.from_pretrained(self.CLAMP_MODEL_NAME)
        self.music_length = self.model.config.max_length
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.io_path = io_path
        
        if query is None or query == "":
            print("WARNING: Query is empty. Only useful for debugging.")
            query = ""
        print(f"Query is: {query}")

        # encode query
        query_ids = self.encoding_data([query], self.QUERY_MODAL)
        self.query_feature = self.get_features(query_ids, self.QUERY_MODAL)

    def encoding_data(self,data,modal):
        """
        Encode the data into ids

        Args:
            data (list): List of strings
            modal (str): "music" or "text"
        
        Returns:
            ids_list (list): List of ids
        """
        ids_list = []
        if modal=="music":
            for item in data:
                patches = CLaMP_Semantic.patchilizer.encode(item, music_length=self.music_length, add_eos_patch=True)
                ids_list.append(torch.tensor(patches).reshape(-1))
        else:
            for item in data:
                text_encodings = CLaMP_Semantic.tokenizer(item, 
                                            return_tensors='pt', 
                                            truncation=True, 
                                            max_length=CLaMP_Semantic.TEXT_LENGTH)
                ids_list.append(text_encodings['input_ids'].squeeze(0))

        return ids_list


    def get_features(self,ids_list, modal):
        """
        Get the features from the CLaMP model

        Args:
            ids_list (list): List of ids
            modal (str): "music" or "text"
        
        Returns:
            features_list (torch.Tensor): Tensor of features with a shape of (batch_size, hidden_size)
        """
        features_list = []
        #print("Extracting "+modal+" features...")
        with torch.no_grad():
            for ids in ids_list: #tqdm(ids_list):
                ids = ids.unsqueeze(0)
                if modal=="text":
                    masks = torch.tensor([1]*len(ids[0])).unsqueeze(0)
                    features = self.model.text_enc(ids.to(self.device), attention_mask=masks.to(self.device))['last_hidden_state']
                    features = self.model.avg_pooling(features, masks)
                    features = self.model.text_proj(features)
                else:
                    masks = torch.tensor([1]*(int(len(ids[0])/PATCH_LENGTH))).unsqueeze(0)
                    features = self.model.music_enc(ids, masks)['last_hidden_state']
                    features = self.model.avg_pooling(features, masks)
                    features = self.model.music_proj(features)

                features_list.append(features[0])
        
        return torch.stack(features_list).to(self.device)


    def zero_shot(self, topn=0):
        print("WARNING: zero_shot() is deprecated. Use zero_shot_fast() instead.")
        # load keys
        keys = []
        key_filenames = []
        # load music keys
        for root, dirs, files in os.walk(self.io_path + "clamp_inference/music_keys"):
            for file in files:
                filename = root+"/"+file
                if filename.endswith(".abc"): # .mxl
                    key_filenames.append(filename)
        #print("Loading music...")

        # load keys if the pth file exists
        if os.path.exists(self.io_path + "clamp_inference/cache/"+self.KEY_MODAL+"_key_cache_"+str(self.music_length)+".pth"):
            with open(self.io_path + "clamp_inference/cache/"+self.KEY_MODAL+"_key_cache_"+str(self.music_length)+".pth", 'rb') as f:
                if self.device == torch.device('cpu'):
                    key_cache = torch.load(f,map_location=self.device)
                else:
                    key_cache = torch.load(f)
            cached_keys = key_cache["keys"]
            cached_key_filenames = key_cache["filenames"]
            cached_key_features = key_cache["features"]
            
            # remove cache that are not in the key_filenames
            files_to_remove = []
            for i, key_filename in enumerate(cached_key_filenames):
                if key_filename not in key_filenames:
                    files_to_remove.append(i)
            
            cached_keys = [key for i, key in enumerate(cached_keys) if i not in files_to_remove]
            cached_key_filenames = [filename for i, filename in enumerate(cached_key_filenames) if i not in files_to_remove]
            cached_key_features = [feature for i, feature in enumerate(cached_key_features) if i not in files_to_remove]

            if len(cached_key_features) > 0:
                cached_key_features = torch.stack(cached_key_features).to(self.device)

            # only keep files that are not in the cache
            key_filenames = [filename for filename in key_filenames if filename not in cached_key_filenames]

        for filename in key_filenames: #tqdm(key_filenames):
            key = unidecode(load_music(filename))
            keys.append(key)  
        non_empty_keys = []
        non_empty_filenames = []

        for key, filename in zip(keys, key_filenames):
            if key.strip()!="":
                non_empty_keys.append(key)
                non_empty_filenames.append(filename)
            else:
                print("File %s not successfully loaded" %(filename))
        
        keys = non_empty_keys
        key_filenames = non_empty_filenames

        # encode keys
        #print(len(keys))
        if len(keys)>0:
            key_ids = CLaMP_Semantic.encoding_data(self,keys, self.KEY_MODAL)
            key_features = self.get_features(key_ids, self.KEY_MODAL)

        # merge cache with new keys
        if os.path.exists(self.io_path + "clamp_inference/cache/"+self.KEY_MODAL+"_key_cache_"+str(self.music_length)+".pth"):
            if len(keys)>0:
                keys = cached_keys + keys
                key_filenames = cached_key_filenames + key_filenames
                if len(cached_key_features)>0:
                    key_features = torch.cat((cached_key_features, key_features), dim=0)
            else:
                keys = cached_keys
                key_filenames = cached_key_filenames
                key_features = cached_key_features
        key_cache = {"keys": keys, "filenames": key_filenames, "features": key_features}
            
        # save key cache as pth file
        if not os.path.exists(self.io_path + "clamp_inference/cache"):
            os.makedirs(self.io_path + "clamp_inference/cache")
        with open(self.io_path +"clamp_inference/cache/"+self.KEY_MODAL+"_key_cache_"+str(self.music_length)+".pth", 'wb') as f:
            torch.save(key_cache, f)

        # compute values
        values = compute_values(self.query_feature, key_features)
        sims = torch.cosine_similarity(self.query_feature, key_features)

        if topn==0 and values.dim() > 0:
            topn = len(values)

        erg = []

        if values.dim() == 0:
            prob = values.item()*100
            sim = sims.item()
            content = key_filenames[0]
            erg.append((sim,content))
            return erg

        for idx in torch.argsort(values)[-topn:]:
            prob = values[idx].item()*100
            sim = sims[idx].item()
            content = key_filenames[idx]
            erg.append((sim,content))
        
        return erg
        
        
    def zero_shot_fast(self, abc_strings=None):
        keys = abc_strings
        
        # encode keys
        #print(len(keys))
        if len(keys)>0:
            key_ids = CLaMP_Semantic.encoding_data(self,keys, self.KEY_MODAL)
            key_features = self.get_features(key_ids, self.KEY_MODAL)

        # compute values
        values = compute_values(self.query_feature, key_features)
        sims = torch.cosine_similarity(self.query_feature, key_features)
        
        return sims.tolist()

        
