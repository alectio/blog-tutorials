'''
query strategy 
'''
import torch
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler

class SubsetSampler(sampler.Sampler):
    '''sample subsets given by indices without random permutation'''
    def __init__(self, indices):
        self.indices = indices
    
    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class EntropySelectQuery(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def __call__(self, unlabeled, n_rec, batch_size=512):
        # infer on the unlabeled set
        dataloader = DataLoader(self.dataset,
            batch_size=batch_size,
            sampler=SubsetSampler(unlabeled))
        
        infered = []
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                infered.append(self.model(imgs))
        
        # stack output into one tensor of shape len(unlabeled) x num_classes
        infered = torch.cat(infered, dim=0)

        # compute entropy
        entropy = self._entropy(infered)

        # select <n_rec> of samples with the highest entropy
        return self._select(unlabeled, entropy, n_rec)

    def _entropy(self, infered_data):
        entropy = -torch.sum(infered_data * torch.log(infered_data), dim=1)
        return entropy.cpu().numpy().tolist()

    def _select(self, samples, metric, n_rec):
        '''select <n_rec> from <samples> with the highest <metric> 
        '''
        selected = [(i, m) for i, m in zip(samples, metric)]

        # sort according to m in descending order
        selected.sort(key=lambda x : -x[1])

        return [x[0] for x in selected[:n_rec]]


