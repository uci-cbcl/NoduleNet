import torch
import numpy as np
class SplitComb():
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value
        
    def split(self, data, side_len = None, max_stride = None, margin = None):
        if side_len==None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin
        
        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))
        self.nz, self.nh, self.nw = nz, nh, nw
        
        nzhw = [nz,nh,nw]
        self.nzhw = nzhw
        
        pad = [ [0, 0],
                [margin, nz * side_len - z + margin],
                [margin, nh * side_len - h + margin],
                [margin, nw * side_len - w + margin]]
        data = np.pad(data, pad, 'edge')

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits,nzhw

    def combine(self, output, nzhw = None, side_len=None, stride=None, margin=None):
        
        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw.any() == None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz,nh,nw = nzhw
        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        side_len /= stride
        margin /= stride

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        output = -1000000 * np.ones((
            nz * side_len,
            nh * side_len,
            nw * side_len,
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1

        return output

    def combine2(self, output, split, nzhw=None, side_len=None, stride=None, margin=None):
        if side_len == None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw == None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz, nh, nw = nzhw
        assert (side_len % stride == 0)
        assert (margin % stride == 0)

        res = []
        idx = 0
        for id in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sd = id * side_len
                    ed = (id + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    origin = np.array([0, sd, sh, sw, 0], dtype=np.float32)
                    origin = origin - np.array([0, margin, margin, margin, 0], dtype=np.float32)

                    _, D, H, W = split[idx].shape
                    s = np.array([margin, margin, margin], dtype=np.float32)
                    e = np.array([min(margin + side_len, D), min(margin + side_len, H), min(margin + side_len, W)])

                    filtered = filter_pbboxes(output[idx], s, e)
                    if len(filtered) > 0:
                        filtered += origin
                        res.append(filtered)

                    idx += 1
        res = np.array(res)
        if len(res) > 0:
            res = np.concatenate(res, 0)
        return res

    def combine3(self, input, nzhw=None, side_len=None, stride=None, margin=None):
        if side_len == None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw == None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz, nh, nw = nzhw
        assert (side_len % stride == 0)
        assert (margin % stride == 0)
        side_len /= stride
        margin /= stride

        _, channels, D, H, W = input.shape
        output = torch.ones((1, channels, nz * side_len, nh * side_len, nw * side_len)).float().cuda()

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = input[idx][:, margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[:, :, sz:ez, sh:eh, sw:ew] = split
                    idx += 1

        return output

def filter_pbboxes(pbboxes, s, e):
    res = []
    for box in pbboxes:
        if np.all(box[1:4] > s) and np.all(box[1:4] < e):
            res.append(box)
    return np.array(res)
