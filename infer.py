import numpy as np
import os
import torchvision.transforms as T

from train import Compose

test_folder = './test'
model.load_state_dict(torch.load('./models'))  

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
model.to(device)
model.eval()

test_files = os.listdir(test_folder)
file_numbers = np.array([int(s[:-4]) for s in test_files])
sort_idxs = np.argsort(file_numbers)

test_files = np.array(test_files)[sort_idxs]


def exchange_coordinate(boxes):
    
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    new_boxes = np.zeros_like(boxes)
    new_boxes[:, 0] = y0
    new_boxes[:, 1] = x0
    new_boxes[:, 2] = y1
    new_boxes[:, 3] = x1


predictions = []
original_sizes = []


test_transform = T.Compose([
    T.Resize(64,64),
    T.ToTensor()
])

for i, fn in enumerate(test_files):
    img = Image.open(os.path.join(test_folder, fn))
    x = test_transform(img).unsqueeze(0)

    pred = model(x.to(device))

    boxes = pred[0]['boxes'].detach().cpu().numpy()
    labels = pred[0]['labels'].detach().cpu().numpy() + 1
    scores = pred[0]['scores'].detach().cpu().numpy()

    predictions += [[boxes, labels, scores]]
    original_sizes += [img.size]
   

for i in range(len(predictions)):
    predictions[i][0] = exchange_coordinate(predictions[i][0])


pred_results = []
threshold = 0.5


for pred in predictions:
    boxes = pred[0]
    labels = pred[1]
    scores = pred[2]

    cond = scores > threshold
    scores = scores[cond]
    boxes = boxes[cond]
    labels = labels[cond]

    prediction = {}
    prediction['bbox'] = boxes.tolist()
    prediction['score'] = scores.tolist()
    prediction['label'] = labels.tolist()

    pred_results += [prediction]


def transform_coords(y0, x0, y1, x1, predicted_size, original_size):
    
    scales = original_size / predicted_size
    new_y0 = round(y0 * scales[1])
    new_x0 = round(x0 * scales[0])
    new_y1 = round(y1 * scales[1])
    new_x1 = round(x1 * scales[0])

    return new_y0, new_x0, new_y1, new_x1


os_pred_results = copy.deepcopy(pred_results)
 
# size of train transformed images
T_size = np.array(64,64)

for i, sample in enumerate(pred_results):
    num_digits = len(sample['bbox'])

    original_size = np.array(original_sizes[i])

    for j in range(num_digits):
        y0, x0, y1, x1 = sample['bbox'][j]
        _y0, _x0, _y1, _x1 = transform_coords(
            y0, x0, y1, x1, T_size, original_size)
        os_pred_results[i]['bbox'][j] = [_y0, _x0, _y1, _x1]


with open('outputs/pred_results.json', 'w') as outfile:
    json.dump(os_pred_results, outfile)