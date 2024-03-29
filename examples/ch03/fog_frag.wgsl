struct LightUniforms {
    lightPosition: vec4f,
    eyePosition: vec4f, 
    specularColor: vec4f,
    fogColor: vec4f,
}
@group(1) @binding(0) var<uniform> light : LightUniforms;

struct MaterialUniforms {
    // blinn-phong
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    // fog
    minDist: f32,
    maxDist: f32,
}
@group(1) @binding(1) var<uniform> material : MaterialUniforms;

struct Input {
    @location(0) vPosition:vec4f, 
    @location(1) vNormal:vec4f, 
    @location(2) vColor: vec4f,
}

fn blinnPhong(N:vec3f, L:vec3f, V:vec3f) -> vec2f {
    let H = normalize(L + V);
    var diffuse = material.diffuse * max(dot(N, L), 0.0);
    diffuse += material.diffuse * max(dot(-N, L), 0.0);
    var specular = material.specular * pow(max(dot(N, H), 0.0), material.shininess);
    specular += material.specular * pow(max(dot(-N, H),0.0), material.shininess);
    return vec2(diffuse, specular);
}

@fragment
fn fs_main(in:Input) ->  @location(0) vec4f {
    var N = normalize(in.vNormal.xyz);                
    let L = normalize(light.lightPosition.xyz - in.vPosition.xyz);     
    let V = normalize(light.eyePosition.xyz - in.vPosition.xyz);     
    let z = abs(in.vPosition.z);
    
    var alpha = 1.0 - (material.maxDist - z)/(material.maxDist - material.minDist);
    alpha = clamp(alpha, 0.0, 1.0);

    let bp = blinnPhong(N, L, V);
                 
    let shaderColor = in.vColor*(material.ambient + bp[0]) + light.specularColor * bp[1]; 
    let finalColor = mix(shaderColor.rgb, light.fogColor.rgb, alpha);
    return vec4(finalColor, 1.0);
}