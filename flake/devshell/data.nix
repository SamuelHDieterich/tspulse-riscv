{
  perSystem =
    { config, pkgs, ... }:
    {
      packages.download = pkgs.python312Packages.callPackage ../../scripts/download/package.nix { };
      packages.series = pkgs.python312Packages.callPackage ../../scripts/series/package.nix { };

      devShells.data = pkgs.mkShell {
        # Exposes only bin/, because the package propagates its dependencies, and Python's setup hook would put them on PYTHONPATH.
        # symlinkJoin doesn't help: it links nix-support/ too, so the propagation survives.
        packages = [
          (pkgs.runCommand "download-bin" { } ''
            mkdir $out
            ln -s ${config.packages.download}/bin $out/bin
          '') # download-datasets, download-weights, download-lists
          (pkgs.runCommand "series-bin" { } ''
            mkdir $out
            ln -s ${config.packages.series}/bin $out/bin
          '') # select-series
        ];
      };
    };
}
