{
  perSystem =
    { config, pkgs, ... }:
    let
      # Newlib's erff and expf, which the host build compiles in so it rounds like rv64 does.
      # fdlibm.h includes machine/ieeefp.h, which glibc has no copy of, and putting newlib's whole libc/include on the path would replace glibc's math.h.
      newlibMath = pkgs.runCommand "newlib-math" { } ''
        mkdir -p $out/src $out/include/machine
        tar -xzf ${pkgs.pkgsCross.riscv64-embedded.newlib.src} -C $out/src --strip-components=1
        cp $out/src/newlib/libc/include/machine/ieeefp.h $out/include/machine/
      '';
    in
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

        # inputsFrom forwards shellHook but not env, so these are exported here.
        shellHook = /* bash */ ''
          export NEWLIB_SRC="${newlibMath}/src"
          export NEWLIB_HOST_INCLUDE="${newlibMath}/include"
        '';
      };
    };
}
