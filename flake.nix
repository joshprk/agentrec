{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    lib = nixpkgs.lib;
    forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    getPkgs = system: import nixpkgs {inherit system;};
  in {
    packages = forAllSystems (system: {
      agentrec = {};

      default = self.packages.${system}.agentrec;
    });

    formatter = forAllSystems (system: let
      pkgs = getPkgs system;
    in
      pkgs.alejandra);
  };
}
