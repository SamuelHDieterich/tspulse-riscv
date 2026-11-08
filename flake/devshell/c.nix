{
  perSystem =
    { config, pkgs, ... }:
    {
      packages.weights = pkgs.python312Packages.callPackage ../../scripts/weights/package.nix { };

      devShells.c = pkgs.mkShell {
        packages = with pkgs; [
          gcc # host C compiler
          gnumake # build driver
          # Exposes only bin/, for the same reason as the download wrapper in data.nix
          (pkgs.runCommand "weights-bin" { } ''
            mkdir $out
            ln -s ${config.packages.weights}/bin $out/bin
          '') # export-weights
        ];
      };
    };
}
