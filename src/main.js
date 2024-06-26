let txId;
let myImage;
let myShader;
let cw, ch;
let cnv;
let seed;

let vertexShader = `
attribute vec3 aPosition;
attribute vec2 aTexCoord;

varying vec2 vTexCoord;

void main() {
  vTexCoord = aTexCoord;
  vec4 positionVec4 = vec4(aPosition, 1.0);
  positionVec4.xy = positionVec4.xy * 2.0 - 1.0;
  gl_Position = positionVec4;
}
`;

let fragmentShader = `
precision mediump float;

uniform vec2 u_resolution;
uniform float u_seed;
uniform sampler2D u_image;
uniform vec3 u_color1;
uniform vec3 u_color2;

varying vec2 vTexCoord;

float random(float x) {
  return fract(sin(x) * 43758.5453);
}

float random2d(vec2 n) {
  return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
  vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec4 invertedColor(vec4 inColor) {
  float centerBrightness = 0.5;
  float powerCurve = 1.0;
  float colorize = 0.0;
  vec3 hsvColor = rgb2hsv(inColor.rgb);

  hsvColor.b = pow(hsvColor.b, powerCurve);
  hsvColor.b = (hsvColor.b < centerBrightness) ? (hsvColor.b / centerBrightness) : (hsvColor.b - centerBrightness) / centerBrightness;
  hsvColor.g = hsvColor.g * hsvColor.b * colorize;

  vec3 outColor = hsv2rgb(hsvColor);

  return vec4(outColor, 1.0);
}

float bayer(int iter, vec2 rc) {
  float sum = 0.0;

  for (int i = 0; i < 4; ++i) {
    float b = float(i < iter);
    vec2 bsize = vec2(pow(2.0, float(i + 1)));
    vec2 t = mod(rc, bsize) / bsize;
    int idx = int(dot(floor(t * 2.0), vec2(2.0, 1.0)));

    float b1 = step(1.0, float(idx)) * 2.0;
    float b2 = step(2.0, float(idx)) * 3.0;
    float b3 = step(3.0, float(idx)) * 1.0;

    b *= b1 + (1.0 - step(1.0, float(idx))) * (b2 + (1.0 - step(2.0, float(idx))) * b3);

    sum += b * pow(4.0, float(iter - i - 1));
  }

  float phi = pow(4.0, float(iter)) + 1.0;
  return (sum + 1.0) / phi;
}

vec4 ditheringColor(vec4 inColor) {
  float luminance = dot(inColor.rgb, vec3(0.2126, 0.7152, 0.0722));
  float threshold = bayer(int(4.0), gl_FragCoord.xy);

  vec3 outColor = mix(vec3(0.0), vec3(1.0), step(threshold, luminance));

  return vec4(outColor, 1.0);
}

float random(vec2 c){
  return fract(sin(dot(c.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float calculateHash(float value) { return fract(sin(value) * 43758.5453); }
float calculateHash(vec2 value) { return fract(sin(dot(value, vec2(12.9898, 64.8975))) * 43758.5453); }
vec2 calculateDualHash(float value) { return vec2(calculateHash(value + 35.28), calculateHash(value + 847.123)); }
vec2 calculateDualHash(vec2 value) { return vec2(calculateHash(value + vec2(58.21)), calculateHash(value + vec2(999.999))); }

vec4 glitchColor(sampler2D u_image, vec2 uv, float u_seed) {
  vec2 baseUv = uv;
  float seed = 0.8147 *  fract(u_seed);

  vec2 transformedUV = fract(uv);

  float randomValue = -1.0;

  for (int i = 0; i < 20; i++) {
    float iterationFloat = float(i) + 1.0;
    float randomOffset = random(vec2(iterationFloat + seed));
    vec2 dualHash = calculateDualHash(iterationFloat + seed);
    transformedUV = baseUv * iterationFloat + dualHash;
    transformedUV *= dualHash * 2.0 - 1.0;

    vec2 integerUV = floor(transformedUV);
    float hashedIntegerUV = calculateHash(integerUV);

    float condition = step(hashedIntegerUV, 0.1);
    baseUv += condition * (integerUV * 2.0 - 1.0);
    randomValue = mix(randomValue, randomOffset, condition);
  }

  vec3 glitchThreshold = vec3(fract(baseUv), randomValue);
  vec3 outColor = mix(
    texture2D(u_image, uv).rgb,
    texture2D(u_image, uv + vec2(0.1 * (glitchThreshold.z * 2.0 - 1.0), 0.0)).rgb,
    step(0.0, glitchThreshold.z)
  );
  return vec4(outColor, 1.0);
}

vec3 normalizeColor(vec3 color) {
  return color / 255.0;
}

vec4 duoColor(vec4 inColor) {
  float gray = dot(inColor.rgb, vec3(0.299, 0.587, 0.114));
  vec3 duoColor1 = u_color1;
  vec3 duoColor2 = u_color2;

  return vec4(mix(duoColor2, duoColor1, gray), 1.0);
}

vec4 blockNoiseColor(sampler2D u_image, vec2 uv, float u_seed) {
  float noiseBaseTime = random2d(vec2(6405.0, 8405.0) + vec2(u_seed * 400.0));

  float bnUvX = uv.x + sin(noiseBaseTime) * 0.1;

  float noiseRandR = random2d(vec2(noiseBaseTime, 8284.0));
  float noiseRandG = random2d(vec2(noiseBaseTime, 9485.0));
  float noiseRandB = random2d(vec2(noiseBaseTime, 5213.0));

  vec2 rnUv = vec2(bnUvX, uv.y);

  vec2 bnRUv;
  vec2 bnGUv;
  vec2 bnBUv;

  float conditionR = step(0.9, noiseRandR);
  float conditionG = step(0.9, noiseRandG);
  float conditionB = step(0.9, noiseRandB);

  bnRUv = mix(rnUv, vec2(bnUvX + noiseRandR, uv.y + noiseRandR), conditionR);
  bnGUv = mix(rnUv, vec2(bnUvX - noiseRandG, uv.y + noiseRandG), conditionR);
  bnBUv = mix(rnUv, vec2(bnUvX - noiseRandB, uv.y - noiseRandB), conditionR);

  bnRUv = mix(bnRUv, vec2(bnUvX - noiseRandR, uv.y - noiseRandR), conditionG * (1.0 - conditionR));
  bnGUv = mix(bnGUv, vec2(bnUvX + noiseRandG, uv.y + noiseRandG), conditionG * (1.0 - conditionR));
  bnBUv = mix(bnBUv, vec2(bnUvX - noiseRandB, uv.y - noiseRandB), conditionG * (1.0 - conditionR));

  bnRUv = mix(bnRUv, vec2(bnUvX - noiseRandR, uv.y - noiseRandR), conditionB * (1.0 - conditionR) * (1.0 - conditionG));
  bnGUv = mix(bnGUv, vec2(bnUvX - noiseRandG, uv.y - noiseRandG), conditionB * (1.0 - conditionR) * (1.0 - conditionG));
  bnBUv = mix(bnBUv, vec2(bnUvX + noiseRandB, uv.y + noiseRandB), conditionB * (1.0 - conditionR) * (1.0 - conditionG));

  float bnR = texture2D(u_image, bnRUv).r * 0.9;
  float bnG = texture2D(u_image, bnGUv).g * 0.9;
  float bnB = texture2D(u_image, bnBUv).b * 0.9;

  return vec4(bnR, bnG, bnB, 1.0);
}

vec4 halfToneColor(vec4 inColor, vec2 uv) {
  vec2 center = vec2(0.9);
  float angle = 0.0;
  float scale = 9.0;
  vec2 dotSize = vec2(80.0);
  float s = sin(angle);
  float c = cos(angle);
  vec2 tex = uv * dotSize - center;
  vec2 point = vec2(c * tex.x - s * tex.y, s * tex.x + c * tex.y) * scale;
  float pattern =  (sin(point.x) * sin(point.y)) * 4.0;

  float average = (inColor.r + inColor.g + inColor.b) / 3.0;
  vec4 outColor = vec4(vec3(average * 10.0 - 5.0 + pattern), inColor.a);

  return outColor;
}

vec4 waveColor(vec4 color, float amount)
{
  color.r = (sin(color.r * amount * 1.0) + 1.0) * 0.4;
  color.g = (sin(color.g * amount * 2.0) + 1.0) * 0.4;
  color.b = (sin(color.b * amount * 4.0) + 1.0) * 0.6;
  return color;
}

vec4 mergeColor(vec4 colors[12], float u_seed) {
  float seed = u_seed;
  vec4 mergedColor;

  if (seed >= 0.0 && seed <= 0.0833) {
    mergedColor = colors[0];
  } else if (seed > 0.0833 && seed <= 0.1667) {
    mergedColor = colors[1];
  } else if (seed > 0.1667 && seed <= 0.25) {
    mergedColor = colors[2];
  } else if (seed > 0.25 && seed <= 0.3333) {
    mergedColor = colors[3];
  } else if (seed > 0.3333 && seed <= 0.4167) {
    mergedColor = colors[4];
  } else if (seed > 0.4167 && seed <= 0.5) {
    mergedColor = colors[5];
  } else if (seed > 0.5 && seed <= 0.5833) {
    mergedColor = colors[6];
  } else if (seed > 0.5833 && seed <= 0.6667) {
    mergedColor = colors[7];
  } else if (seed > 0.6667 && seed <= 0.75) {
    mergedColor = colors[8];
  } else if (seed > 0.75 && seed <= 0.8333) {
    mergedColor = colors[9];
  } else if (seed > 0.8333 && seed <= 0.9167) {
    mergedColor = colors[10];
  } else if (seed > 0.9167 && seed <= 1.0) {
    mergedColor = colors[11];
  } else {
    mergedColor = vec4(0.0);
  }

  return mergedColor;
}

float direction(float n) {
  float threshold = random(n);
  return threshold > 0.5 ? 1.0 : -1.0;
}

void main() {
  vec2 uv = gl_FragCoord.xy/u_resolution.xy;
  uv.y = 1.0 - uv.y;

  vec4 initialColor = texture2D(u_image, uv);

  vec4 glitchColor1 = glitchColor(u_image, uv, u_seed / 0.2);
  vec4 glitchColor2 = glitchColor(u_image, uv, u_seed * 0.2);
  vec4 glitchColor3 = glitchColor(u_image, uv, u_seed / 0.3);
  vec4 glitchColor4 = glitchColor(u_image, uv, u_seed * 0.3);
  vec4 glitchColor5 = glitchColor(u_image, uv, u_seed * 0.5);
  vec4 glitchColor6 = glitchColor(u_image, uv, u_seed * 0.5);

  vec4 waveColor1 = waveColor(glitchColor1, 10.0 * u_seed);
  vec4 waveColor2 = waveColor(glitchColor2, 20.0 * u_seed);
  vec4 waveColor3 = waveColor(glitchColor3, 15.0 * u_seed);
  vec4 waveColor4 = waveColor(glitchColor4, 8.0 * u_seed);
  vec4 waveColor5 = waveColor(glitchColor5, 12.0 * u_seed);
  vec4 waveColor6 = waveColor(glitchColor6, 12.0 * u_seed);

  vec4 invertedColor1 = invertedColor(glitchColor1);
  vec4 invertedColor2 = invertedColor(glitchColor2);
  vec4 invertedColor3 = invertedColor(glitchColor3);
  vec4 invertedColor4 = invertedColor(glitchColor4);
  vec4 invertedColor5 = invertedColor(glitchColor5);
  vec4 invertedColor6 = invertedColor(glitchColor6);

  vec4 duoColor1 = duoColor(glitchColor1);
  vec4 duoColor2 = duoColor(glitchColor2);
  vec4 duoColor3 = duoColor(glitchColor3);
  vec4 duoColor4 = duoColor(glitchColor4);
  vec4 duoColor5 = duoColor(glitchColor5);
  vec4 duoColor6 = duoColor(glitchColor6);

  vec4 halfToneColor1 = halfToneColor(glitchColor1, uv);
  vec4 halfToneColor2 = halfToneColor(glitchColor2, uv);
  vec4 halfToneColor3 = halfToneColor(glitchColor3, uv);
  vec4 halfToneColor4 = halfToneColor(glitchColor4, uv);
  vec4 halfToneColor5 = halfToneColor(glitchColor5, uv);
  vec4 halfToneColor6 = halfToneColor(glitchColor6, uv);

  vec4 ditheringColor1 = ditheringColor(glitchColor1);
  vec4 ditheringColor2 = ditheringColor(glitchColor2);
  vec4 ditheringColor3 = ditheringColor(glitchColor3);
  vec4 ditheringColor4 = ditheringColor(glitchColor4);
  vec4 ditheringColor5 = ditheringColor(glitchColor5);
  vec4 ditheringColor6 = ditheringColor(glitchColor6);

  vec4 invertedDitheringColor1 = ditheringColor(invertedColor1);

  vec4 colors[12];

  colors[0] = waveColor1 + duoColor2 * ditheringColor3 + glitchColor4 * halfToneColor5;
  colors[1] = invertedColor1 + ditheringColor2 / glitchColor3 * waveColor4 + fract(duoColor5 * 0.5 + initialColor);
  colors[2] = halfToneColor1 * duoColor2 + waveColor3 + 0.5 + abs(ditheringColor4 + glitchColor5);
  colors[3] = duoColor1 + invertedColor2 / halfToneColor3 + abs(glitchColor4 + invertedColor5 - 0.1);
  colors[4] = ditheringColor1 + duoColor2 + halfToneColor3 * invertedColor5 + abs(initialColor);
  colors[5] = waveColor1 * ditheringColor2 + duoColor3 + invertedColor4 + fract(initialColor * 0.003);
  colors[6] = invertedDitheringColor1 / waveColor2 * duoColor3 + fract(glitchColor4 * 0.05);
  colors[7] = (initialColor / ditheringColor2 * duoColor3) + waveColor4 * fract(halfToneColor5);
  colors[8] = (duoColor1 + waveColor2 + halfToneColor3 + waveColor4) + glitchColor5;
  colors[9] = invertedColor1 * 0.5 / duoColor2 * waveColor3 + ditheringColor4 + glitchColor5;
  colors[10] = ditheringColor1 * waveColor2 / duoColor3 + glitchColor4 * halfToneColor5;
  colors[11] = invertedColor1 * duoColor2 + waveColor3 * ditheringColor4;

  vec4 outColor = mergeColor(colors, u_seed);

  float luminance = dot(outColor.rgb, vec3(0.2126, 0.7152, 0.0722));
  float highThreshold = 0.8;
  float lowThreshold = 0.2;
  vec3 adjustment = vec3(0.6);
  vec3 lessThanAdjustment = adjustment * step(luminance, lowThreshold);
  vec3 greaterThanAdjustment = adjustment * step(highThreshold, luminance);
  outColor.rgb = outColor.rgb + greaterThanAdjustment - lessThanAdjustment;

  gl_FragColor = outColor;
}
`;

const darkColorCodes = [
  '6B6C6E',
  '34365E',
  '696EBF',
  '0843F4',
  '4159FE',
  '43A6B8',
  'F2837B',
  'E97B16',
  'F46436',
  'E50008',
  'B55606',
  'B426C4',
  'A806E7',
  '5528AA',
  '4E75B7',
  '4D61DC'
];

function selectRandomValues(arr, num) {
  const result = [];
  const cloneArr = [...arr];

  for (let i = 0; i < num; i++) {
    const randomIndex = Math.floor(random() * cloneArr.length);
    const selectedImage = cloneArr.splice(randomIndex, 1)[0];
    result.push(selectedImage);
  }
  return result;
}

function hexToRgb(hex) {
  let bigint = parseInt(hex, 16);
  let r = (bigint >> 16) & 255;
  let g = (bigint >> 8) & 255;
  let b = bigint & 255;

  let normalizedR = r / 255.0;
  let normalizedG = g / 255.0;
  let normalizedB = b / 255.0;

  normalizedR = parseFloat(normalizedR.toFixed(4));
  normalizedG = parseFloat(normalizedG.toFixed(4));
  normalizedB = parseFloat(normalizedB.toFixed(4));

  return [normalizedR, normalizedG, normalizedB];
}

function fetchSeed(txId) {
  let id = txId;
  if (!id || id === "") {
    return 115338;
  }
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    let char = id.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0;
  }
  let seed = Math.abs(hash) % 1000000;
  return seed;
}

function canvasSize() {
  let w = windowWidth;
  let h = windowHeight;
  if (w >= h) {
    cw = h * myImage.width / myImage.height;
    ch = h;
  } else {
    cw = w;
    ch = w * myImage.height / myImage.width;
  }
  return [cw, ch];
}

function setup(){
  seed = fetchSeed(txId);
  randomSeed(seed);

  pixelDensity(1);
  cnv = createCanvas(...canvasSize(), WEBGL);

  centerCanvas();
  myShader = createShader(vertexShader, fragmentShader);
  shader(myShader);
  noStroke();
  seedValue = random();
  colorCodes = darkColorCodes;
  colors = selectRandomValues(colorCodes, 2);
  color1 = hexToRgb(colors[0]);
  color2 = hexToRgb(colors[1]);
}

function draw(){
  drawShader();
}

function drawShader(){
  let yMouse = mouseY / height;
  let xMouse = mouseX / width;
  myShader.setUniform('u_resolution', [width, height]);
  myShader.setUniform('u_image', myImage);
  myShader.setUniform('u_seed', seedValue);
  myShader.setUniform('u_color1', color1);
  myShader.setUniform('u_color2', color2);

  rect(0,0,width,height);
}

function windowResized() {
  resizeCanvas(...canvasSize());
  centerCanvas();
}

function centerCanvas() {
  let x = (windowWidth - width) / 2;
  let y = (windowHeight - height) / 2;
  cnv.position(x, y);
}

function keyTyped() {
  if (key == 's') {
    let prevWidth = width;
    let prevHeight = height;

    resizeCanvas(2666, 4000);
    drawShader();

    saveCanvas('artwork', 'png');
    resizeCanvas(prevWidth, prevHeight);
  }
}
