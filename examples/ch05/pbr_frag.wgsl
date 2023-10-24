const pi = 3.14159265358979323846;

struct LightUniforms {
    lightPosition : vec4f,
    lightIntensity : vec4f,
};
@group(1) @binding(0) var<uniform> light:array<LightUniforms, 4>;

struct MaterialUniforms {
    eyePosition : vec4f, 
    roughness: f32, 
    metallic: f32,
};
@group(1) @binding(1) var<uniform> material : MaterialUniforms;

fn ggxDistrbution(dotNh: f32) -> f32 {
    let a = material.roughness * material.roughness * material.roughness * material.roughness;
    let b = (dotNh * dotNh) * (a - 1.0) + 1.0;
    return a / (pi * b * b);
}

fn geomSmith(dotProd: f32) -> f32 {
    let a = (material.roughness + 1.0) * (material.roughness + 1.0) /8.0;
    return 1.0 / (dotProd * (1.0 - a) + a);
}

fn schlickFresnel(dotHv:f32, vColor:vec4f) -> vec3f {
    var f0 = vec3(0.04);
    f0 = mix(f0, vColor.rgb, material.metallic);
    return f0 + (1.0 - f0) * pow(clamp(1.0 - dotHv, 0.0, 1.0), 5.0);
}

fn microfacetModel(idx: u32, position: vec3f, n: vec3f, vColor:vec4f) -> vec3f {
    var l = normalize(light[idx].lightPosition.xyz - position);
    let v = normalize(material.eyePosition.xyz - position);
    let h = normalize( v + l);
    let dist = length(light[idx].lightPosition.xyz - position);
    var intensity = light[idx].lightIntensity.xyz;
    intensity /= (dist * dist);
    let dotNh = dot(n, h);
    let dotLh = dot(l, h);
    let dotNl = max(dot(n, l), 0.0);
    let dotNv = max(dot(n, v), 0.0);
    let dotHv = dot(h, v);

    let ndf = ggxDistrbution(dotNh);
    let g = geomSmith(dotNv) * geomSmith(dotNl);
    let f = schlickFresnel(dotHv, vColor);

    let specularBrdf = 0.25 * ndf * g * f / (0.0001 + dotNv * dotNl);
    
    let ks = f;
    var kd = vec3(1.0) - ks;
    kd *= 1.0 - material.metallic;
    
    return (kd *vColor.rgb / pi + specularBrdf) * intensity * dotNl;
}

struct Input {
    @location(0) vPosition:vec4f, 
    @location(1) vNormal:vec4f, 
    @location(2) vColor:vec4f,
}

@fragment
fn fs_main(in:Input) ->  @location(0) vec4f {
    var color = vec3(0.0);
    var n = normalize(in.vNormal.xyz);   
    for(var i = 0u; i < 4u; i = i + 1u){
        color += microfacetModel(i, in.vPosition.xyz, n, in.vColor);
        color += microfacetModel(i, in.vPosition.xyz, -n, in.vColor);
    }

    // Reinhard operator
    color = color / (color + vec3(1.0));
    // gamma correction
    color = pow(color, vec3(1.0/2.2));

    return vec4(color, 1.0);
}