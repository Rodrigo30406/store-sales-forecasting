import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--register",
        nargs="?",
        const="None",
        choices=["None", "Staging", "Production"],
        help="Registra el modelo. Usa 'Staging' o 'Production' para asignar etapa, o nada para dejarlo sin etapa."
    )
    return parser.parse_args()