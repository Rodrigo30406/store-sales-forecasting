from src.training import prepare_data, train_model_lgbm, evaluate_model
from src.utils import save_model
from src.config import CONFIG

def main():
    X_train, X_val, y_train, y_val = prepare_data()
    model = train_model_lgbm(X_train, X_val, y_train, y_val)
    save_model(model, CONFIG.get_path("model"))
    print(f"✅ Entrenamiento finalizado y modelo guardado en ",CONFIG.get_path("model"))
    results = evaluate_model(model, X_val, y_val)
    print("✅ Evaluación:", results)
    
if __name__ == "__main__":
    main()
