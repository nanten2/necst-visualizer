import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from astropy.coordinates import SkyCoord
from astropy.coordinates import AltAz, FK5, Galactic
import xarray as xr
import numpy as np
from datetime import datetime
from typing import Union, Optional, Tuple

import necstdb
import n_const.constants as n2const

PathLike = Union[str, Path]
timestamp2datetime = np.vectorize(datetime.utcfromtimestamp)


class ScanCheck:
    """
    Tool to check whether the data is taken with "normally" controlled
    system.
    """

    ENCODER_TOPIC = "status_encoder"
    OBSMODE_TOPIC = "obsmode"

    def __init__(self, data_path: PathLike, kisa_path: PathLike = None) -> None:
        self.data_path = Path(data_path)
        # self.kisa_path = Path(kisa_path)
        self.db = necstdb.opendb(self.data_path)
        self.encoder_data = self.load_data(self.ENCODER_TOPIC)
        self.obsmode_data = self.load_data(self.OBSMODE_TOPIC)
        self.target = str(self.data_path).split("_")[-1]

    def load_data(self, topic_name: str) -> dict:
        data = self.db.open_table(topic_name).read(astype="array")
        return {field: data[field] for field in data.dtype.names}

    def create_data_array(self, dump: bool = False) -> xr.Dataset:
        encoder_array = (
            xr.Dataset({k: xr.DataArray(v) for k, v in self.encoder_data.items()})
            .rename({"dim_0": "t", "timestamp": "t"})
            .set_coords("t")
            .swap_dims({"t": "t"})
        )

        coords = self.trans_coords(encoder_array)

        obsmode_array = (
            xr.Dataset({k: xr.DataArray(v) for k, v in self.obsmode_data.items()})
            .rename({"dim_0": "t", "received_time": "t"})
            .set_coords("t")
            .swap_dims({"t": "t"})
        )

        drive_data = encoder_array.assign(
            obsmode_array.reindex_like(encoder_array, "nearest")
        ).assign_coords(coords)
        if dump is True:
            pickle_dir = "./"
        if dump:
            pickle_path = Path(pickle_dir) / Path(self.data_path.stem).with_suffix(
                ".pickle"
            )
            with open(pickle_path, "wb") as f:
                pickle.dump(drive_data, f)
            return pickle_path.absolute()
        return drive_data

    def trans_coords(self, enc: xr.Dataset) -> dict:
        """
        Transform azel coordinate to equatorial and galactic coordinates.
        Notes
        -----
        This function will take 100s to process data of length 300k.
        """

        print(
            "Calculating coordinates, "
            f"will take about {enc.sizes.get('t') / 3000:.0f} s"
        )
        # empirical, about 3000 data are processed per second

        _az = enc.enc_az / 3600
        _el = enc.enc_el / 3600

        # if self.kisa_path:
        #     d_az, d_el = apply_kisa_test(azel=(_az, _el), hosei=self.kisa_path)
        # else:
        #     d_az, d_el = 0, 0
        d_az, d_el = 0, 0

        az = _az + d_az / 3600
        el = _el + d_el / 3600

        coord_horizontal = SkyCoord(
            az=az.data,
            alt=el.data,
            frame=AltAz,
            location=n2const.LOC_NANTEN2,
            obstime=timestamp2datetime(enc.t),
            unit="deg",
        )
        # If you give the coordinate as xr.DataArray, it may take 100 times longer
        # to get astropy object.

        coord_equatorial = coord_horizontal.transform_to(
            FK5
        )  # this transformation take quite long time
        coord_galactic = coord_equatorial.transform_to(Galactic)

        ret = {
            "az": ("t", az),
            "el": ("t", el),
            "ra": ("t", coord_equatorial.ra),
            "dec": ("t", coord_equatorial.dec),
            "l": ("t", coord_galactic.l),
            "b": ("t", coord_galactic.b),
        }
        return ret


class VisualizeScan:

    MAIN_OBSMODES = [b"ON        ", b"OFF       ", b"SKY       "]
    OTHER_OBSMODES = [b"          ", b"Non       ", b"HOT       "]
    PLOT_COLOR = {
        b"HOT       ": "#F00",
        b"Non       ": "#000",
        b"OFF       ": "#0DF",
        b"SKY       ": "#0DF",
        b"ON        ": "#0F0",
        b"          ": "#777",
    }
    COORD_MAP = {
        "horizontal": {
            "xy": ["az", "el"],
            "title": "Horizontal",
            "label": ["Az. [deg]", "El. [deg]"],
        },
        "equatorial": {
            "xy": ["ra", "dec"],
            "title": "Equatorial (J2000)",
            "label": ["R.A. [deg]", "Dec. [deg]"],
        },
        "galactic": {
            "xy": ["l", "b"],
            "title": "Galactic",
            "label": ["$l$ [deg]", "$b$ [deg]"],
        },
    }

    def __init__(
        self, drive_data: xr.Dataset, observation: str = "NotSpecified"
    ) -> None:
        self.drive_data = drive_data
        self.observation = observation
        self.target = observation.split("_")[-1]
        self.fig, self.ax = plt.subplots(3, 1, figsize=(14, 10))

    @classmethod
    def from_pickle(cls, pickle_path: PathLike) -> None:
        observation = str(pickle_path).split(".")[0]
        with open(pickle_path, "rb") as p:
            return cls(pickle.load(p), observation)

    @classmethod
    def draw_data_centric(cls, ax: matplotlib.axes._axes.Axes) -> None:
        return NotImplemented

    def draw_one_coord(
        self, coord: str = "galactic", fig=None, ax: matplotlib.axes._axes.Axes = None
    ) -> None:
        """
        Parameters
        ----------
        coord: str
            Either of ["horizontal", "equatorial", "galactic"].
        ax: axes object of matplotlib
            Axis to which the data is drawn.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        coord = coord.lower()

        existing_obsmodes = np.unique(self.drive_data.obs_mode)
        # get common obsmodes
        main_obsmodes = set(self.MAIN_OBSMODES).intersection(existing_obsmodes)
        other_obsmodes = set(self.OTHER_OBSMODES).intersection(existing_obsmodes)

        x, y = self.COORD_MAP[coord]["xy"]

        for mode in main_obsmodes:
            self.draw(mode, ax, x, y, zorder=-2)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        for mode in other_obsmodes:
            self.draw(mode, ax, x, y, zorder=-3)

        ax.set(
            title=self.COORD_MAP[coord]["title"],
            xlim=xlim,
            ylim=ylim,
            xlabel=self.COORD_MAP[coord]["label"][0],
            ylabel=self.COORD_MAP[coord]["label"][1],
        )

        if coord != "horizontal":
            _ = [ax.invert_xaxis() for _ in range(1) if ax.xaxis_inverted()]
            ax.invert_xaxis()

        ax.set_rasterization_zorder(0)
        ax.legend()
        ax.grid(True)

        return (fig, ax)

    def track_one_coord(
        self, coord: str = "galactic", fig=None, ax: matplotlib.axes._axes.Axes = None, interval: int = 100
    ) -> None:
        """
        Parameters
        ----------
        coord: str
            Either of ["horizontal", "equatorial", "galactic"].
        ax: axes object of matplotlib
            Axis to which the data is drawn.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        coord = coord.lower()

        existing_obsmodes = np.unique(self.drive_data.obs_mode)
        # get common obsmodes
        main_obsmodes = set(self.MAIN_OBSMODES).intersection(existing_obsmodes)
        # other_obsmodes = set(self.OTHER_OBSMODES).intersection(existing_obsmodes)

        x, y = self.COORD_MAP[coord]["xy"]

        for mode in main_obsmodes:
            self.track(mode, ax, x, y, zorder=-2, interval=interval)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        ax.set(
            title=self.COORD_MAP[coord]["title"],
            xlim=xlim,
            ylim=ylim,
            xlabel=self.COORD_MAP[coord]["label"][0],
            ylabel=self.COORD_MAP[coord]["label"][1],
        )

        if coord != "horizontal":
            _ = [ax.invert_xaxis() for _ in range(1) if ax.xaxis_inverted()]
            ax.invert_xaxis()

        ax.set_rasterization_zorder(0)
        ax.grid(True)

        return (fig, ax)

    def draw(self, mode, ax, x, y, zorder):
        mod = mode.decode("utf8")
        settings = {
            "s": 0.1,
            "zorder": zorder,
            "label": mod,
            "c": self.PLOT_COLOR[mode],
        }
        drive_dat = self.drive_data.where(self.drive_data.obs_mode == mode, drop=True)
        ax.scatter(drive_dat[x], drive_dat[y], **settings)

    def track(self, mode, ax, x, y, zorder, interval: int = 100):
        drive_dat = self.drive_data.where(self.drive_data.obs_mode == mode, drop=True)
        x_track = [
            drive_dat[x][i * interval]
            for i in range(round(len(drive_dat[x]) / interval))
        ]
        y_track = [
            drive_dat[y][i * interval]
            for i in range(round(len(drive_dat[y]) / interval))
        ]
        dx = [x_track[i + 1] - x_track[i] for i in range(len(x_track) - 1)]
        dy = [y_track[i + 1] - y_track[i] for i in range(len(y_track) - 1)]
        dx.append(np.nan)
        dy.append(np.nan)
        len(dx), len(dy)
        ax.quiver(
            x_track,
            y_track,
            dx,
            dy,
            scale=1,
            scale_units="xy",
            angles="xy",
            alpha=0.5,
            zorder=zorder,
        )

    def draw_figure(
        self, save: Union[PathLike, bool] = False, fig=None, axes=None
    ) -> Optional[Path]:
        print("Drawing a figure...")
        if axes is None:
            fig, axes = self.fig, self.ax

        coord_list = ["horizontal", "equatorial", "galactic"]
        for i in range(3):
            self.draw_one_coord(coord=coord_list[i], ax=axes[i])

        plt.tight_layout()

        if save is True:
            save_dir = "./"
        if save:
            fig_path = Path(save_dir) / f"{self.target}_observation.pdf"
            plt.savefig(fig_path, dpi=150)
            return fig_path.absolute()

        return (fig, axes)

    def track_figure(
        self, save: Union[PathLike, bool] = False, fig=None, axes=None, interval: int = 100
    ) -> Optional[Path]:
        print("Drawing a figure...")
        if axes is None:
            fig, axes = self.fig, self.ax

        coord_list = ["horizontal", "equatorial", "galactic"]
        for i in range(3):
            self.track_one_coord(coord=coord_list[i], ax=axes[i], interval=interval)

        plt.tight_layout()

        if save is True:
            save_dir = "./"
        if save:
            fig_path = Path(save_dir) / f"{self.target}_observation.pdf"
            plt.savefig(fig_path, dpi=150)
            return fig_path.absolute()

        return (fig, axes)
