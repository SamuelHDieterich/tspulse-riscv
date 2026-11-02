{
  perSystem =
    { pkgs, ... }:
    let
      target = pkgs.pkgsCross.riscv64-embedded;
    in
    {
      devShells.riscv = pkgs.mkShell {
        packages = with pkgs; [
          spike # RISC-V ISA simulator
          dtc # device tree compiler, which Spike runs at startup
        ];

        # Neither of these can go in packages:
        # the cross GCC's setup hook would point CC, LD, AR and others at the RISC-V tools, breaking host builds.
        # pk would be spliced to its build-platform variant, which does not exist for x86_64.
        # Spike does not search PATH for pk, so it runs as `spike "$RISCV_PK" program.elf`.
        shellHook = /* bash */ ''
          export PATH="${target.buildPackages.gcc}/bin:$PATH"
          export RISCV_PK="${target.riscv-pk}/bin/pk"
        '';
      };
    };
}
