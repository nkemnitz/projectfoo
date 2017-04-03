import BABYLON from 'babylonjs';

export default
class DatasetParser implements BABYLON.Internals.CustomTextureParser {
  gl: WebGL2RenderingContext;
  height: number;
  width: number;
  depth: number;
  target: number;
  level: number;
  internalFormat: number;
  border: number;
  format: number;
  type: number;
  src: any;

  constructor(engine: any, arrayBuffer: ArrayBuffer) {
    this.gl = engine._gl as WebGL2RenderingContext;
    this.level = 0;
    this.internalFormat = this.gl.RGBA8UI;
    this.height = 4;
    this.width = 4;
    this.depth = 4;
    this.border = 0;
    this.format = this.gl.RGBA_INTEGER;
    this.type = this.gl.UNSIGNED_BYTE;
    this.src = new Uint8Array(arrayBuffer);
    this.target = (this.depth === void 0) ?
      this.gl.TEXTURE_2D : this.gl.TEXTURE_3D;
  }

  upload(): void {
    this.gl.texParameteri(
      this.target, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(
      this.target, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(
      this.target, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(
      this.target, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    if (this.target === this.gl.TEXTURE_2D) {
      this.gl.texImage2D(this.target, this.level, this.internalFormat,
                         this.width, this.height, this.border, this.format,
                         this.type, this.src);
    } else {
      this.gl.texParameteri(
        this.target, this.gl.TEXTURE_WRAP_R, this.gl.CLAMP_TO_EDGE);
      this.gl.texImage3D(this.target, this.level, this.internalFormat,
                         this.width, this.height, this.depth, this.border,
                         this.format, this.type, this.src);
    }
  }
}
