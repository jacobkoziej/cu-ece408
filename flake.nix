{
  description = "The Cooper Union - ECE 408: Wireless Communications";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    inputs:

    inputs.flake-utils.lib.eachDefaultSystem (
      system:

      let
        pkgs = import inputs.nixpkgs {
          inherit system;
        };

      in
      {
        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
