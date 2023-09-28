import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

def compare_images(img1, img2, cmap='afmhot'):
    fig, axes = plt.subplots(1,3,figsize=(9,4))
    for ax, img in zip(axes[:-1], [img1, img2]):
        im = ax.imshow(img, cmap=cmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.formatter.set_powerlimits((0, 0))
    im = axes[-1].imshow(np.abs(img1-img2), cmap='jet')
    axes[-1].get_xaxis().set_visible(False)
    axes[-1].get_yaxis().set_visible(False)
    axes[-1].set_title('absolute difference')
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.formatter.set_powerlimits((0, 0))
    
    return fig, axes

def slider(movie, fov=1, t_axis=0, ax=None, cmap=None, vmax=None):
    from ipywidgets import interact

    if movie.ndim != 3:
        raise AttributeError('Movie dimensions ({}) different than 3'.format(movie.ndim))

    num_frames = movie.shape[t_axis]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    extent = [-fov/2, fov/2, -fov/2, fov/2]
    im = ax.imshow(np.take(movie, 0, axis=t_axis), extent=extent, origin='lower', cmap=cmap, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)

    def imshow_frame(frame):
        img = np.take(movie, frame, axis=t_axis)
        im.set_array(img)
        cbar.mappable.set_clim([img.min(), img.max()])
        
    interact(imshow_frame, frame=(0, num_frames-1));
    
def animate_movies_synced(movie_list, axes, t_axis=0, fov=1.0, vmin=None, vmax=None, cmaps='afmhot', add_ticks=False,
                   add_colorbars=True, titles=None, fps=10, output=None, flipy=False, bitrate=1e6, dynamic_clim=False):
    
    # Image animation function (called sequentially)
    def animate_frame(i):
        for movie, im, cbar in zip(movie_list, images, cbars):
            img = np.take(movie, i, axis=t_axis)
            im.set_array(img)
            if dynamic_clim:
                cbar.mappable.set_clim([img.min(), img.max()])
        return images

    fig = plt.gcf()
    num_frames, nx, ny = movie_list[0].shape
    extent = [-fov/2, fov/2, -fov/2, fov/2]
        
    # initialization function: plot the background of each frame
    images = []
    cbars = []
    titles = [None]*len(movie_list) if titles is None else titles
    cmaps = [cmaps]*len(movie_list) if isinstance(cmaps, str) else cmaps
    vmin_list = [movie.min() for movie in movie_list] if vmin is None else vmin
    vmax_list = [movie.max() for movie in movie_list] if vmax is None else vmax

    for movie, ax, title, cmap, vmin, vmax in zip(movie_list, axes, titles, cmaps, vmin_list, vmax_list):
        if add_ticks == False:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap)
        if add_colorbars:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbars.append(fig.colorbar(im, cax=cax))
        im.set_clim(vmin, vmax)
        images.append(im)
        if flipy:
            ax.invert_yaxis()

    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate_frame, frames=num_frames, interval=1e3 / fps)

    if output is not None:
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
        anim.save(output, writer=writer)
    return anim