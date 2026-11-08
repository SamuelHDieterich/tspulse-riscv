{
  perSystem =
    { config, pkgs, ... }:
    {
      packages.weights = pkgs.python312Packages.callPackage ../../scripts/weights/package.nix { };

      devShells.c = pkgs.mkShell {
        packages = with pkgs; [
          gcc # host C compiler
          gnumake # build driver
          # Exposes only bin/, because the package propagates its dependencies, and Python's setup hook would put them on PYTHONPATH
          (pkgs.runCommand "weights-bin" { } ''
            mkdir $out
            ln -s ${config.packages.weights}/bin $out/bin
          '') # export-weights
        ];
      };
    };
}
