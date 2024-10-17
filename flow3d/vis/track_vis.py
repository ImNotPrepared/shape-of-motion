




  @torch.no_grad()
  def vis_tracks_2d_video(
      path,
      imgs: np.ndarray,
      tracks_3d: np.ndarray,
      Ks: np.ndarray,
      w2cs: np.ndarray,
      occs=None,
      radius: int = 3,
  ):
      num_tracks = tracks_3d.shape[0]
      labels = np.linspace(0, 1, num_tracks)
      cmap = get_cmap("gist_rainbow")
      colors = cmap(labels)[:, :3]
      tracks_2d = (
          project_2d_tracks(tracks_3d.swapaxes(0, 1), Ks, w2cs).cpu().numpy()  # type: ignore
      )
      frames = np.asarray(
          draw_keypoints_video(imgs, tracks_2d, colors, occs, radius=radius)
      )
      iio.imwrite(path, frames, fps=15)