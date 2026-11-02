{ inputs, ... }:
{
  imports = [ inputs.flake-file.flakeModules.dendritic ];

  flake-file.description = "TSPulse on RISC-V: edge anomaly detection";

  # The dendritic preset assumes a ./modules root.
  flake-file.outputs = /* nix */ ''
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (inputs.import-tree ./flake)
  '';

  systems = [ "x86_64-linux" ];
}
