import os
import json
import time
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "/home/claude/dataset_repo/data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS_TOTAL = 8

MODEL_PATH = "/home/claude/cat_dog_model.keras"       # resume checkpoint (latest weights)
BEST_MODEL_PATH = "/home/claude/cat_dog_model_best.keras"  # best-val-accuracy weights (served to app)
STATE_PATH = "/home/claude/train_state.json"
CLASS_NAMES_PATH = "/home/claude/class_names.json"
TRAIN_CACHE = "/home/claude/tfcache_train"
VAL_CACHE = "/home/claude/tfcache_val"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"epoch": 0, "epochs_total": EPOCHS_TOTAL, "status": "not_started",
            "history": [], "best_val_accuracy": 0.0}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def build_model():
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.06)(x)
    x = layers.RandomZoom(0.1)(x)

    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        return x

    x = block(x, 32)
    x = block(x, 64)
    x = block(x, 128)
    x = block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.1, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.1, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
    )
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache(TRAIN_CACHE).shuffle(2000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache(VAL_CACHE).prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def run_final_test(model):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)
    loss, acc = model.evaluate(test_ds, verbose=0)
    return float(loss), float(acc)


def main():
    state = load_state()
    epoch = state["epoch"]

    if epoch >= state["epochs_total"]:
        log("Training already complete.")
        return

    log(f"Preparing epoch {epoch + 1}/{state['epochs_total']}")
    train_ds, val_ds, class_names = get_datasets()

    if not os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "w") as f:
            json.dump(class_names, f)

    if epoch == 0 or not os.path.exists(MODEL_PATH):
        log("Building fresh model.")
        model = build_model()
    else:
        log("Loading model checkpoint.")
        model = tf.keras.models.load_model(MODEL_PATH)
        if epoch >= 4:
            new_lr = 3e-4
            model.optimizer.learning_rate.assign(new_lr)
            log(f"Lowered learning rate to {new_lr} for stability.")

    state["status"] = "training"
    save_state(state)

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2)
    elapsed = time.time() - t0

    record = {
        "epoch": epoch + 1,
        "loss": float(history.history["loss"][0]),
        "accuracy": float(history.history["accuracy"][0]),
        "val_loss": float(history.history["val_loss"][0]),
        "val_accuracy": float(history.history["val_accuracy"][0]),
        "seconds": round(elapsed, 1),
    }
    log(f"Epoch {epoch + 1} done in {elapsed:.1f}s: {record}")

    improved = record["val_accuracy"] > state.get("best_val_accuracy", 0.0)
    model.save(MODEL_PATH)  # always save latest, so we can resume
    if improved:
        state["best_val_accuracy"] = record["val_accuracy"]
        model.save(BEST_MODEL_PATH)
        log("New best val_accuracy — saved best model checkpoint.")

    state["epoch"] = epoch + 1
    state["history"].append(record)

    if state["epoch"] >= state["epochs_total"]:
        log("All epochs complete. Running final test evaluation on best checkpoint...")
        best_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
        best_model = tf.keras.models.load_model(best_path)
        test_loss, test_acc = run_final_test(best_model)
        state["status"] = "done"
        state["test_loss"] = test_loss
        state["test_accuracy"] = test_acc
        log(f"Final test accuracy: {test_acc:.4f}")
    else:
        state["status"] = "epoch_complete"

    save_state(state)
    log("State saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        state = load_state()
        state["status"] = "error"
        state["error"] = str(e)
        state["traceback"] = traceback.format_exc()
        save_state(state)
        raise
