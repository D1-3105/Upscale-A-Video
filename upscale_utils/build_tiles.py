import math
from typing import NamedTuple

import torch


class TileInfo(NamedTuple):
    x: int
    y: int
    input_start_x: int
    input_end_x: int
    input_start_y: int
    input_end_y: int
    input_start_x_pad: int
    input_start_y_pad: int


class TilesBuilder:
    def __init__(self, tile_side_size: int, input_shape: torch.Size, tile_overlap: int = 64):
        # output shape = (b, c, t, output_h, output_w)
        b, c, t, h, w = input_shape
        self.tile_height = self.tile_width = tile_side_size
        self.tile_overlap_height = self.tile_overlap_width = tile_overlap
        self.output_shape = (b, c, t, h * 4, w * 4)
        self.output = torch.Tensor(
            (),
        ).new_zeros(self.output_shape, dtype=torch.uint8)
        self.tiles_x = math.ceil(w / self.tile_width)
        self.tiles_y = math.ceil(h / self.tile_height)
        self.rm_end_pad_w, self.rm_end_pad_h = True, True

        if (self.tiles_x - 1) * self.tile_width + self.tile_overlap_width >= w:
            self.tiles_x = self.tiles_x - 1
            self.rm_end_pad_w = False

        if (self.tiles_y - 1) * self.tile_height + self.tile_overlap_height >= h:
            self.tiles_y = self.tiles_y - 1
            self.rm_end_pad_h = False

    def flush_output(self):
        self.output = torch.Tensor(
            (),
        ).new_zeros(self.output_shape, dtype=torch.uint8)

    def add_processed_tile(self, output_tile, tile_info: TileInfo):
        x, y, input_start_x, input_end_x, input_start_y, input_end_y, input_start_x_pad, input_start_y_pad = tile_info
        input_tile_width = input_end_x - input_start_x
        input_tile_height = input_end_y - input_start_y
        output_start_x = input_start_x * 4
        if x == self.tiles_x - 1 and not self.rm_end_pad_w:
            output_end_x = self.output_shape[-1]
        else:
            output_end_x = input_end_x * 4

        output_start_y = input_start_y * 4
        if y == self.tiles_y - 1 and not self.rm_end_pad_h:
            output_end_y = self.output_shape[-2]
        else:
            output_end_y = input_end_y * 4

        # output tile area without padding
        output_start_x_tile = (input_start_x - input_start_x_pad) * 4
        if x == self.tiles_x - 1 and not self.rm_end_pad_w:
            output_end_x_tile = output_start_x_tile + self.output_shape[-1] - output_start_x
        else:
            output_end_x_tile = output_start_x_tile + input_tile_width * 4
        output_start_y_tile = (input_start_y - input_start_y_pad) * 4
        if y == self.tiles_y - 1 and not self.rm_end_pad_h:
            output_end_y_tile = output_start_y_tile + self.output_shape[-2] - output_start_y
        else:
            output_end_y_tile = output_start_y_tile + input_tile_height * 4

        # put tile into output image
        self.output[:, :, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
            :, :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
        ]

    def gen_tiles(self, vframes: torch.Tensor, flows_bi: list[torch.Tensor]):
        tile_width, tile_height = self.tile_width, self.tile_height
        tile_overlap_width, tile_overlap_height = self.tile_overlap_width, self.tile_overlap_height
        _, _, _, h, w = vframes.shape
        tiles_x = self.tiles_x
        tiles_y = self.tiles_y

        print(f"Processing the video w/ tile patches [{tiles_x}x{tiles_y}]...")  # noqa
        for y in range(tiles_y):
            for x in range(tiles_x):
                print(f"\ttile: [{y + 1}/{tiles_y}] x [{x + 1}/{tiles_x}]")  # noqa
                # extract tile from input image
                ofs_x = x * tile_width
                ofs_y = y * tile_height
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_width, w)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_height, h)
                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_overlap_width, 0)
                input_end_x_pad = min(input_end_x + tile_overlap_width, w)
                input_start_y_pad = max(input_start_y - tile_overlap_height, 0)
                input_end_y_pad = min(input_end_y + tile_overlap_height, h)

                input_tile = vframes[:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                if flows_bi:
                    flows_bi_tile = [
                        flows_bi[0][:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad],
                        flows_bi[1][:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad],
                    ]
                else:
                    flows_bi_tile = None
                yield (
                    input_tile,
                    flows_bi_tile,
                    TileInfo(
                        x,
                        y,
                        input_start_x,
                        input_end_x,
                        input_start_y,
                        input_end_y,
                        input_start_x_pad,
                        input_start_y_pad,
                    ),
                )
