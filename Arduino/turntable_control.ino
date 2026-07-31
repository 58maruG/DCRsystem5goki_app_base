/* ============================================================
 * サクランボ病害虫鳥獣被害果除去システム 弐号機
 * ターンテーブル制御 — Arduino UNO版
 *
 * ラズパイ + pigpio のDMA波形制御を、ATmega328P の Timer1
 * ハードウェアタイマーに置き換えたもの。
 * PULパルスをCPUを使わずハードウェアで生成するため、
 * タイミングが正確でジッタがなく、異音が出にくい。
 *
 * 動作モードはハイブリッド（D6の物理スイッチで切替）:
 *   - PCモード  : PCのGUIから USBシリアル経由でコマンド受信
 *   - 単独モード : PCなしで、非常停止スイッチの開閉だけで回転
 *
 * Arduinoが非同期に送る通知行（PC側が受信して処理する）:
 *   - ESTOP / ESTOP_CLEARED : 非常停止の作動 / 解除
 *   - STANDALONE / PC_MODE   : 単独モードへ移行 / PCモードへ復帰
 *     ※モード切替時に送る。起動時点で既に単独モードなら READY 直後にも
 *       STANDALONE を送り、PC側のロック状態を同期する。
 *
 * 非常停止スイッチ(オムロン製・c接点を2ブロック)は2回路使う:
 *   - 回路1: c接点のNC側をTB6600のVccに直列（押すと電源遮断＝確実な物理停止）
 *   - 回路2: 空きブロックのNC接点をD5に接続（押すと開=Arduinoが検知してPC通知・LED表示）
 * モーターを止める安全経路は回路1のVcc物理遮断。回路2は通知・表示用。
 * ============================================================ */

// --- ピン割り当て -------------------------------------------
// PUL_PIN は Timer1 の OC1A に固定で割り当てられているため、
// D9 から変更できない（ハードウェア出力のため）。
const uint8_t PUL_PIN   = 9;   // ステップパルス出力 (Timer1 OC1A 固定)
const uint8_t DIR_PIN   = 8;   // 回転方向
const uint8_t ESTOP_PIN = 5;   // 非常停止 入力 (内部プルアップ) ※24V短絡事故でD3が故障したためD5へ移設
const uint8_t MODE_PIN  = 6;   // モード切替 入力 (LOW=PC / HIGH=単独, 内部プルアップ) ※D4故障のためD6へ移設
const uint8_t LED_PIN   = 7;   // 状態LED 出力 ※D5を非常停止に使うためD7へ移設

// 非常停止スイッチの接点種別 (D5に何をつないだか)。
//   false = NC(b接点): 通常=閉(LOW)  / 押下=開(HIGH)   ← 今回はこれ(実配線がNC接点。フェイルセーフ)
// 配線変更時はここだけ変えればよい。
const bool ESTOP_NC = false;

// --- 速度テーブル -------------------------------------------
// ラズパイ版 SPEED_MAP と同じHIGH/LOW幅[マイクロ秒]。
// レベルが大きいほど周期が短く ＝ 回転が速い。
const uint16_t SPEED_DELAY_US[10] = {
  1000, // 1: 遅い
   900, // 2
   800, // 3
   700, // 4
   600, // 5
   500, // 6
   400, // 7
   300, // 8
   200, // 9
   100  // 10: 速い
};

// --- 状態 ---------------------------------------------------
uint8_t currentSpeed = 6;        // 1〜10
bool    isRunning    = false;
bool    estopActive  = false;    // 非常停止が作動中か（ロックアウト）
bool    singleMode   = false;    // true=単独モード, false=PCモード

// --- 入力デバウンス -----------------------------------------
// 機械接点のチャタリングで誤検知・通知の連発をしないよう、
// 一定時間(DEBOUNCE_MS)安定してから状態を確定する。
const unsigned long DEBOUNCE_MS = 20;

bool          estopRaw, estopStable;          // ピン生値 / 確定値 (true=作動)
unsigned long estopChangedAt;
bool          modeRaw, modeStable;            // ピン生値 / 確定値 (true=単独)
unsigned long modeChangedAt;

// --- LED点滅用 ----------------------------------------------
unsigned long ledBlinkAt = 0;
bool          ledBlinkState = false;
const unsigned long LED_BLINK_MS = 300;

// --- シリアル受信バッファ -----------------------------------
char    cmdBuf[16];
uint8_t cmdLen = 0;

// ============================================================
// 指定したHIGH/LOW幅[us]を作るための OCR1A 値を計算する。
//   プリスケーラ N=8、F_CPU=16MHz のとき
//   トグル間隔[us] = (OCR1A + 1) * 8 / 16 = (OCR1A + 1) / 2
//   → OCR1A = 2 * delay_us - 1
// ============================================================
static inline uint16_t calcOCR(uint16_t delay_us) {
  return (uint16_t)(2UL * delay_us - 1);
}

// ============================================================
// 回転開始: Timer1 を CTCモードで起動し、OC1A(D9) をトグル出力。
// パルス生成はハードウェアが行うため、loop()の負荷とは無関係に
// 正確な周波数を出し続ける。
// 非常停止中は安全のため起動させない。
// ============================================================
void startRotation() {
  if (isRunning || estopActive) return;

  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;
  OCR1A  = calcOCR(SPEED_DELAY_US[currentSpeed - 1]);

  TCCR1A |= (1 << COM1A0);   // コンペアマッチごとに OC1A(D9) をトグル
  TCCR1B |= (1 << WGM12);    // CTCモード (TOP = OCR1A)
  TCCR1B |= (1 << CS11);     // プリスケーラ 8 でタイマー始動
  interrupts();

  isRunning = true;
}

// ============================================================
// 回転停止: タイマーを止め、PULピンを LOW に固定する。
// ============================================================
void stopRotation() {
  noInterrupts();
  TCCR1B = 0;                 // クロック停止
  TCCR1A = 0;                 // OC1A を通常ピンに戻す
  interrupts();
  digitalWrite(PUL_PIN, LOW);
  isRunning = false;
}

// ============================================================
// 速度変更: OCR1A を書き換える。回転中もシームレスに反映される。
// ============================================================
void setSpeed(uint8_t level) {
  if (level < 1 || level > 10) return;
  currentSpeed = level;

  if (isRunning) {
    uint16_t newOCR = calcOCR(SPEED_DELAY_US[currentSpeed - 1]);
    noInterrupts();
    OCR1A = newOCR;
    // 現在のカウントが新しいTOPを超えていると、一度65535まで
    // 数え上げてから折り返してしまう。その場合だけ0に戻して
    // 即座に新しい周期へ切り替える。
    if (TCNT1 > newOCR) TCNT1 = 0;
    interrupts();
  }
}

// ============================================================
// 1行ぶんのコマンドを解釈して実行する。（PCモード時のシリアル）
//   R       : 回転開始
//   S       : 停止
//   V<1-10> : 速度変更 (例 V7)
//   C       : クリーンアップ(停止・速度を基準に初期化)
//   P       : ピング（PC側の接続確認用。OK:PONGを返すだけ）
//   Q       : 状態問い合わせ（現在の非常停止状態を ESTOP/ESTOP_CLEARED で返す）
// ============================================================
void handleCommand(const char* cmd) {
  switch (cmd[0]) {
    case 'R':
      if (estopActive) {
        Serial.println(F("ERR:ESTOP"));   // 非常停止中は起動拒否
        break;
      }
      startRotation();
      Serial.println(F("OK:ROTATE"));
      break;

    case 'S':
      stopRotation();
      Serial.println(F("OK:STOP"));
      break;

    case 'V': {
      int level = atoi(cmd + 1);
      if (level >= 1 && level <= 10) {
        setSpeed((uint8_t)level);
        Serial.print(F("OK:SPEED="));
        Serial.println(level);
      } else {
        Serial.println(F("ERR:SPEED_RANGE"));
      }
      break;
    }

    case 'C':
      stopRotation();
      currentSpeed = 6;
      Serial.println(F("OK:CLEANUP"));
      break;

    case 'P':
      Serial.println(F("OK:PONG"));
      break;

    case 'Q':
      // PC起動時の状態同期用。現在の非常停止状態を通知トークンで返す。
      Serial.println(estopActive ? F("ESTOP") : F("ESTOP_CLEARED"));
      break;

    default:
      Serial.println(F("ERR:UNKNOWN_CMD"));
      break;
  }
}

// ============================================================
// デバウンス付きで入力ピンを読む。
//   raw を更新し、DEBOUNCE_MS 以上同じなら stable を確定する。
//   stable が変化したとき true を返す（エッジ検出用）。
// ============================================================
bool debounceRead(uint8_t pin, bool &raw, bool &stable,
                  unsigned long &changedAt, unsigned long now) {
  bool v = (digitalRead(pin) == HIGH);
  if (v != raw) {            // 生値が変わった → 計測リセット
    raw = v;
    changedAt = now;
  }
  if (raw != stable && (now - changedAt) >= DEBOUNCE_MS) {
    stable = raw;            // 安定したので確定
    return true;             // 変化あり
  }
  return false;
}

// ============================================================
// 状態LEDの更新（ブロックしないmillisベース）。
//   非常停止中 : 点滅 / 回転中 : 点灯 / 停止中 : 消灯
// ============================================================
void updateLed(unsigned long now) {
  if (estopActive) {
    if (now - ledBlinkAt >= LED_BLINK_MS) {
      ledBlinkAt = now;
      ledBlinkState = !ledBlinkState;
      digitalWrite(LED_PIN, ledBlinkState ? HIGH : LOW);
    }
  } else {
    digitalWrite(LED_PIN, isRunning ? HIGH : LOW);
  }
}

// ============================================================
void setup() {
  pinMode(PUL_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(ESTOP_PIN, INPUT_PULLUP);
  pinMode(MODE_PIN,  INPUT_PULLUP);   // LOW=PCモード / HIGH=単独モード

  digitalWrite(PUL_PIN, LOW);
  digitalWrite(DIR_PIN, HIGH);         // 回転方向。逆にしたい場合は HIGH に
  digitalWrite(LED_PIN, LOW);

  // 起動時の入力状態を確定値として読み込んでおく
  unsigned long now = millis();
  estopRaw = estopStable = (digitalRead(ESTOP_PIN) == HIGH);
  modeRaw  = modeStable  = (digitalRead(MODE_PIN)  == HIGH);
  estopChangedAt = modeChangedAt = now;
  // ピンHIGH→押下か否かは接点種別で解釈する (NO: LOWで押下 / NC: HIGHで押下)
  estopActive = ESTOP_NC ? !estopStable : estopStable;
  singleMode  = modeStable;

  Serial.begin(115200);
  Serial.println(F("READY"));   // PC側はこの行で起動完了を判定できる
  // 起動時点で既に単独モードなら、PC側と状態を同期するため通知しておく。
  if (singleMode) Serial.println(F("STANDALONE"));
}

// ============================================================
void loop() {
  unsigned long now = millis();

  // --- モード切替の監視 --------------------------------------
  // モードが変わったら安全側に一旦停止する。
  // modeJustChanged を立てて、同イテレーション内での auto-start を防ぐ。
  bool modeJustChanged = false;
  if (debounceRead(MODE_PIN, modeRaw, modeStable, modeChangedAt, now)) {
    singleMode = modeStable;
    stopRotation();
    modeJustChanged = true;
    // PCへモード変化を通知する。PC側は単独モード中、GUIの操作系をロックする。
    Serial.println(singleMode ? F("STANDALONE") : F("PC_MODE"));
  }

  // --- 非常停止の監視（エッジ検出） --------------------------
  if (debounceRead(ESTOP_PIN, estopRaw, estopStable, estopChangedAt, now)) {
    // ピンHIGH→押下かは接点種別で解釈 (NO: LOWで押下 / NC: HIGHで押下)
    bool pressed = ESTOP_NC ? !estopStable : estopStable;
    if (pressed && !estopActive) {
      // 作動: 即停止してロックアウトし、PCへ通知
      estopActive = true;
      stopRotation();
      Serial.println(F("ESTOP"));
    } else if (!pressed && estopActive) {
      // 解除: ロックアウト解除してPCへ通知
      estopActive = false;
      Serial.println(F("ESTOP_CLEARED"));
      // PCモードは再指令まで停止維持（ここでは回さない）。
      // 単独モードは後段のレベル制御で自動再開する。
    }
  }

  // --- 単独モードのレベル制御 --------------------------------
  // 非常停止が解除されている間は回し続ける（操作系は非常停止のみ）。
  // モード切替直後は1イテレーション待ってから再起動する（即時再起動防止）。
  if (singleMode && !estopActive && !isRunning && !modeJustChanged) {
    startRotation();
  }

  // --- PCモードのシリアル処理 --------------------------------
  // シリアルを1文字ずつ読み、改行までを1コマンドとして処理する。
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdLen > 0) {
        cmdBuf[cmdLen] = '\0';
        handleCommand(cmdBuf);
        cmdLen = 0;
      }
    } else if (cmdLen < sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    }
    // バッファ上限に達した場合は、改行が来るまでの超過分を捨てる
    // （cmdLen は増やさないので上書き衝突は起きない）
  }

  // --- 状態LED ----------------------------------------------
  updateLed(now);
}
