import React from 'react';
import BABYLON from 'babylonjs';
import {Cuckoo, DataType} from '../tools/data/CuckooHash';
import DatasetParser from '../tools/data/DatasetParser';
import R16TextureParser from '../tools/data/R16TextureParser';
import RGBA8x512TextureParser from '../tools/data/RGBA8x512TextureParser';
import RGBA8x4TextureParser from '../tools/data/RGBA8x4TextureParser';
import fragmentShader from '../shaders/dvr.fragment.glsl';
import vertexShader from '../shaders/dvr.vertex.glsl';
import composeFragmentShader from '../shaders/compose.fragment.glsl';
import UInt64 from '../../types/uint64';

export interface SceneProps {
  width: number;
  height: number;
  canvasId: string;
}

export interface SceneState {
}

export default class Scene extends React.Component<SceneProps, SceneState> {
  public static propTypes = {
    width: React.PropTypes.number.isRequired,
    height: React.PropTypes.number.isRequired,
    canvasId: React.PropTypes.string,
  };

  public canvas: HTMLCanvasElement;
  public engine: BABYLON.Engine;
  public scene: BABYLON.Scene;
  public camera: BABYLON.ArcRotateCamera;
  public materials: Map<String, BABYLON.ShaderMaterial>;
  public gl: WebGL2RenderingContext;

  private dataset = {
    dimensions: new BABYLON.Vector3(2048, 2048, 2048),
    resolution: new BABYLON.Vector3(4.0, 4.0, 40.0),
  };

  public updateCamera(): void {
    if (this.camera.radius < 0.01) {
      this.camera.radius = 0.01;
    } else if (this.camera.radius > 1000.0) {
      this.camera.radius = 1000.0;
    }
  }

  public componentDidMount() {

    const options: { [id: string]: boolean; } = {
      'alpha': true,
      'depth': true,
      'stencil': false,
      'antialias': false,
      'failIfMajorPerformanceCaveat': true,
    };

    let canvas = document.getElementById(this.props.canvasId) as HTMLCanvasElement;

    if (!canvas) {
      return;
    }

    this.engine = new BABYLON.Engine(canvas, options.antialias, options, false);
    this.materials = new Map<String, BABYLON.ShaderMaterial>();
    this.gl = this.engine._gl as WebGL2RenderingContext;
    this.canvas = canvas;

    this.createScene();

    this.engine.runRenderLoop(() => {
      this.scene.render();
      this.updateCamera();
    });
  }

  public createScene(): void {
    // Scene
    this.scene = new BABYLON.Scene(this.engine);

    // Camera
    this.camera = new BABYLON.ArcRotateCamera('EyeCamera', 0.0, 0.0, 0.0, BABYLON.Vector3.Zero(), this.scene);
    this.camera.setPosition(new BABYLON.Vector3(0.0, 0.0, -1.0));
    this.camera.mode = BABYLON.Camera.PERSPECTIVE_CAMERA;
    this.camera.fov = Math.PI / 4.0;
    this.camera.attachControl(this.canvas, false, true);
    this.camera.maxZ = 1000.0;
    this.camera.minZ = 0.01;
    this.camera.wheelPrecision = 1000.0;
    this.camera.panningSensibility = 10000.0;

    // Dataset
    const physicalExtent = this.dataset.dimensions.multiply(this.dataset.resolution);
    const maxExtent = Math.max(physicalExtent.x, Math.max(physicalExtent.y, physicalExtent.z));
    const distort = new BABYLON.Vector3(maxExtent, maxExtent, maxExtent).divide(physicalExtent);

    // Shader Materials
    let shaderStore = BABYLON.Effect.ShadersStore as { [id: string]: string };
    shaderStore.dvrVertexShader = vertexShader.trim();
    shaderStore.dvrPixelShader = fragmentShader.trim();

    let frontplaneMaterial: BABYLON.ShaderMaterial = new BABYLON.ShaderMaterial('singlePass', this.scene, 'dvr', {
      needAlphaBlending: true,
      attributes: ['uv'],
      uniforms: ['view', 'worldViewProjection'],
    });
    this.materials.set('singlePass', frontplaneMaterial);

    // Mesh
    const frontplane = BABYLON.MeshBuilder.CreatePlane('billboard', { size: 1.0 }, this.scene);
    frontplane.position = new BABYLON.Vector3(0.0, 0.0, 0.0);
    frontplane.material = frontplaneMaterial;

    // Textures + Framebuffers

    // Voxel cache
    const voxelCache = new BABYLON.Texture('datasets/voxelcache.raw', this.scene, true, false,
      BABYLON.Texture.NEAREST_SAMPLINGMODE, undefined, undefined, undefined, false, undefined, R16TextureParser);

    // Page table cache
    const pageTable = new BABYLON.Texture('datasets/pagetable.raw', this.scene, true, false,
      BABYLON.Texture.NEAREST_SAMPLINGMODE, undefined, undefined, undefined, false, undefined, RGBA8x512TextureParser);

    // Multi-resolution page directory
    const pageDirectory = new BABYLON.Texture('datasets/pagedirectory.raw', this.scene, true, false,
      BABYLON.Texture.NEAREST_SAMPLINGMODE, undefined, undefined, undefined, false, undefined, RGBA8x4TextureParser);

    // Segmentation
    const cubeTex = new BABYLON.Texture('datasets/e2198.raw', this.scene, true, false,
      BABYLON.Texture.NEAREST_SAMPLINGMODE, undefined, undefined, undefined, false, undefined, DatasetParser);

    // Selected Segment Hash Texture
    const selectionHash = new Cuckoo(this.scene, DataType.UInt64, 'identity', 'fnv1a');
    /*for (let i = 500; i < 1000; ++i) {
      selectionHash.set(new UInt64(i, 0));
    }
    for (let i = 3000; i < 3500; ++i) {
      selectionHash.set(new UInt64(i, 0));
    }*/
    selectionHash.set(new UInt64(4366, 0));
    selectionHash.set(new UInt64(9247, 0));
    /*for (let i = 0; i < 35000; ++i) {
       selectionHash.set(i, 0);
    }*/

    // RTT Example - final color, segmentID and segmentDepth
    this.gl.activeTexture(this.gl.TEXTURE0);
    const segColorTex = new BABYLON.RenderTargetTexture('segColorTex',
      { width: this.canvas.width, height: this.canvas.height }, this.scene, false, true,
      BABYLON.Engine.TEXTURETYPE_UNSIGNED_INT, false, BABYLON.Texture.NEAREST_SAMPLINGMODE, false, false);

    this.gl.activeTexture(this.gl.TEXTURE1);
    const segIDTex = this.createScreenTexture('segIDTex', this.gl.R32UI, this.gl.RED_INTEGER, this.gl.UNSIGNED_INT);

    this.gl.activeTexture(this.gl.TEXTURE2);
    const depthTex = this.createScreenTexture('segDepthTex', this.gl.RGBA32F, this.gl.RGBA, this.gl.FLOAT);

    this.gl.activeTexture(this.gl.TEXTURE0);

    const fb = segColorTex._texture._framebuffer;
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fb);

    this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT1, this.gl.TEXTURE_2D,
      segIDTex._texture, 0);
    this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT2, this.gl.TEXTURE_2D,
      depthTex._texture, 0);

    this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0, this.gl.COLOR_ATTACHMENT1, this.gl.COLOR_ATTACHMENT2]);

    if (this.gl.checkFramebufferStatus(this.gl.FRAMEBUFFER) !== this.gl.FRAMEBUFFER_COMPLETE) {
      console.error('Framebuffer creation failed!');
    }

    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);

    // Can't use glClear because of integer buffer, have to write own clear function
    segColorTex.onClear = (engine: BABYLON.Engine) => {
      this.gl.clearBufferfv(this.gl.COLOR, 0, [0.2, 0.2, 0.3, 1.0]); // segColorTex (Segment Color)
      this.gl.clearBufferuiv(this.gl.COLOR, 1, [0, 0, 0, 0]);        // segIDTex (Segment ID)
      this.gl.clearBufferfv(this.gl.COLOR, 2, [0.0, 0.0, 0.0, 0.0]); // depthTex (Segment Depth)
    };

    this.scene.customRenderTargets.push(segColorTex);
    segColorTex.renderList.push(frontplane);

    // Uniforms
    frontplaneMaterial.setVector3('distortionCorrection', distort);
    frontplaneMaterial.setFloat('fovy', this.camera.fov);
    frontplaneMaterial.setVector2('seeds', selectionHash.getSeeds());
    frontplaneMaterial.setVector3('sizes', selectionHash.getSizes());
    frontplaneMaterial.setTexture('cubeTex', cubeTex);
    frontplaneMaterial.setTexture('selectionTex', selectionHash.texture);
    frontplaneMaterial.setTexture('voxelCache', voxelCache);
    frontplaneMaterial.setTexture('pageTable', pageTable);
    frontplaneMaterial.setTexture('pageDirectory', pageDirectory);


    // Post Processes
    shaderStore.composePixelShader = composeFragmentShader.trim();
    const postProcess = new BABYLON.PostProcess('compose', 'compose', [], ['segColorTex', 'segIDTex', 'segDepthTex'],
      1.0, this.camera, BABYLON.Texture.NEAREST_SAMPLINGMODE, this.engine, true);

    postProcess.onApply = (effect) => {
      effect.setTexture('segColorTex', segColorTex);
      effect.setTexture('segIDTex', segIDTex);
      effect.setTexture('segDepthTex', depthTex);
    };
  }

  public componentWillUnmount() {
    this.engine.stopRenderLoop();
    if (this.engine.scenes.length !== 0) {
      while (this.engine.scenes.length > 0) {
        this.engine.scenes[0].dispose();
      }
    }
    this.engine.wipeCaches();
    this.engine.dispose();
  }

  render() {
    return <div>
      <canvas height={this.props.height} width={this.props.width} id={this.props.canvasId} />
    </div>;
  };

  private createScreenTexture(name: string, internalformat: number, format: number, type: number): BABYLON.Texture {
    const babTex = new BABYLON.Texture('', this.scene, true);
    babTex.name = name;
    babTex.isRenderTarget = true;
    babTex.wrapU = BABYLON.Texture.CLAMP_ADDRESSMODE;
    babTex.wrapV = BABYLON.Texture.CLAMP_ADDRESSMODE;
    babTex.coordinatesMode = BABYLON.Texture.PROJECTION_MODE;
    babTex._texture = this.engine.createDynamicTexture(this.canvas.width, this.canvas.height, false,
      BABYLON.Texture.NEAREST_SAMPLINGMODE);

    if (type === this.gl.FLOAT) {
      babTex._texture.type = BABYLON.Engine.TEXTURETYPE_FLOAT;
    } else if (type === this.gl.UNSIGNED_BYTE || type === this.gl.UNSIGNED_SHORT || type === this.gl.UNSIGNED_INT) {
      babTex._texture.type = BABYLON.Engine.TEXTURETYPE_UNSIGNED_INT;
    }

    this.gl.bindTexture(this.gl.TEXTURE_2D, babTex._texture);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, internalformat, this.canvas.width, this.canvas.height, 0, format, type,
      undefined);
    this.gl.bindTexture(this.gl.TEXTURE_2D, null);

    babTex._texture.isReady = true;
    return babTex;
  }
}
