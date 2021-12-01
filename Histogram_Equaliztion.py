from typing import List, Tuple
from PIL import Image
from math import floor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('AGG')


def generate_hist_data(img: np.ndarray) -> dict:
    hist_dict = {}
    for w in range(img.shape[0]):
        for h in range(img.shape[1]):
            if img[w, h] in hist_dict:
                hist_dict[img[w, h]] += 1
            else:
                hist_dict[img[w, h]] = 1

    return hist_dict


def plt_histGraph(data: List[int], savePath: str = None) -> None:
    n, bins, patches = plt.hist(data, bins=256)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    if savePath is not None:
        plt.savefig(savePath)
    else:
        plt.savefig('./out/hist_grpah.png')
    plt.clf()


def histEqualization(img: np.ndarray) -> np.array:
    new_img = np.zeros(img.shape, dtype=np.uint8)  # generate new_img, new_img.shape == img.shape
    hist_dict = generate_hist_data(img)  # count every value's freq.
    sum_items = 0
    for value in sorted(hist_dict.keys()):  # from the lessest value to the largest
        sum_items += hist_dict[value]
        new_img[np.where(img == value)] = np.uint8(
            round(255 * sum_items / img.size)
        )  # find every index that has same value with current value, then use algorithm result to replace it.

    return new_img


def local_histEqualization(img: np.ndarray, kernel: Tuple[int]) -> np.array:
    center = kernel[0] / 2, kernel[1] / 2  # calu. center
    extend_side = floor(center[0]), floor(center[1])  # extend how many space from each side
    extend_img = np.zeros(
        (img.shape[0] + extend_side[0] * 2, img.shape[1] + extend_side[1] * 2), dtype=np.uint8
    )  # generate an extend img
    extend_img[
        extend_side[0] : extend_side[0] + img.shape[0], extend_side[1] : extend_side[1] + img.shape[1]
    ] = img  # put img into the center of extend_img

    # use reflect to padding, if kernel is odd, then reflect from second nearest space (row, col), if is not, then reflect from the nearest space (row, col).
    if center[0] != extend_side[0]:
        extend_img[:, : extend_side[0]] = extend_img[:, extend_side[0] + 1 : extend_side[0] * 2 + 1][:, ::-1]
        extend_img[:, -extend_side[0] :] = extend_img[:, -extend_side[0] * 2 - 1 : -extend_side[0] - 1][:, ::-1]
    else:
        extend_img[:, : extend_side[0]] = extend_img[:, extend_side[0] : extend_side[0] * 2][:, ::-1]
        extend_img[:, -extend_side[0] :] = extend_img[:, -extend_side[0] * 2 : -extend_side[0]][:, ::-1]

    if center[1] != extend_side[1]:
        extend_img[0 : extend_side[1], :] = extend_img[extend_side[1] + 1 : extend_side[1] * 2 + 1, :][::-1]
        extend_img[-extend_side[1] :, :] = extend_img[-extend_side[1] * 2 - 1 : -extend_side[1] - 1, :][::-1]
    else:
        extend_img[0 : extend_side[1], :] = extend_img[extend_side[1] : extend_side[1] * 2, :][::-1]
        extend_img[-extend_side[1] :, :] = extend_img[-extend_side[1] * 2 : -extend_side[1], :][::-1]
    # print(extend_img)

    new_img = np.zeros(img.shape, dtype=np.uint8)  # generate new_img, new_img.shape == img.shape
    for w in range(img.shape[0]):
        for h in range(img.shape[1]):
            block = histEqualization(
                img=extend_img[w : w + extend_side[0] * 2, h : h + extend_side[1] * 2]
            )  # calu. local histogram_equaliztion result

            # find the center value, the put it into new_img[w,h]
            if center[0] != extend_side[0] and center[1] != extend_side[1]:
                new_img[w, h] = block[extend_side[0], extend_side[1]]
            elif center[0] == extend_side[0] and center[1] != extend_side[1]:
                new_img[w, h] = block[extend_side[0] : extend_side[0] + 1, extend_side[1]].mean()
            elif center[0] != extend_side[0] and center[1] == extend_side[1]:
                new_img[w, h] = block[extend_side[0], extend_side[1] : extend_side[1] + 1].mean()
            else:
                new_img[w, h] = block[extend_side[0] : extend_side[0] + 1, extend_side[1] : extend_side[1] + 1].mean()

    return new_img


if __name__ == '__main__':
    from submodules.FileTools.FileSearcher import get_filenames

    paths = get_filenames(dir_path='./TestImg', specific_name='*.tif')  # find all the files in the ./TestImg

    for path in paths:  # execute by each file
        filename = path.split('/')[2].split('.')[0]  # find the filename, e.g. lena, child, etc.
        img = np.array(Image.open(path))

        plt_histGraph(img.reshape(-1), savePath=f'./out/{filename}_hist.png')  # plt histogram graph and save it.
        Image.fromarray(histEqualization(img)).save(
            f'./out/{filename}_HistEqual.png'
        )  # execute global histogram_equalization algorithm and save it.

        kernel = (21, 21)
        Image.fromarray(local_histEqualization(img, kernel)).save(
            f'./out/{filename}_HistEqual-local-{kernel[0]}x{kernel[1]}.png'
        )  # execute local histogram_equalization algorithm and save it.

    # img = np.random.randint((5), size=(7, 7), dtype=np.uint8)
    # print(img)
    # local_histEqualization(img, (4, 4))
