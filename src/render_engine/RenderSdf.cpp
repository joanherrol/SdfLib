#include "RenderSdf.h"
#include "Window.h"
#include "Camera.h"
#include "SdfLib/utils/PrimitivesFactory.h"
#include "shaders/Shader.h"
#include "SdfLib/OctreeSdf.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <spdlog/spdlog.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "shaders/ScreenPlaneShader.h"
#include <imgui.h>

using namespace sdflib;

RenderSdf::~RenderSdf()
{
    glDeleteProgram(mRenderProgramId);
}

void RenderSdf::restart()
{
    glDeleteProgram(mRenderProgramId);
    start();
}

void RenderSdf::start()
{
    auto checkForOpenGLErrors = []() -> GLenum
    {
        GLenum errorCode;
        while((errorCode = glGetError()) != GL_NO_ERROR)
        {
            SPDLOG_ERROR("OpenGL error with code {}", errorCode);
            return errorCode;
        }

        return GL_NO_ERROR;
    };

    // Load shader
    {
        unsigned int computeShaderId = glCreateShader(GL_COMPUTE_SHADER);

        auto loadShaderFromFile = [](std::string path, unsigned long* length) -> char*
        {
            std::ifstream file;
            file.open(path, std::ios_base::in | std::ios_base::binary);
            if (!file.good()) return nullptr;
            file.seekg(0, std::ios::end);
            *length = file.tellg();
            (*length)++;
            char* ret = new char[*length];
            file.seekg(0, std::ios::beg);
            file.read(ret, *length);
            file.close();
            ret[(*length) - 1] = 0;
            return ret;
        };

        unsigned long length;
        char* fileShader = loadShaderFromFile("./shaders/sdfOctreeRender.comp", &length);
        if (fileShader == nullptr) {
            std::filesystem::path p("sdfOctreeRender.comp");
            fileShader = loadShaderFromFile("../src/render_engine/shaders/" + p.filename().string(), &length);
            if (fileShader == nullptr)
                std::cout << "File " << "sdfOctreeRender.comp" << " not found" << std::endl;
        }

        glShaderSource(computeShaderId, 1, &fileShader, NULL);
        glCompileShader(computeShaderId);

        int success;
        glGetShaderiv(computeShaderId, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(computeShaderId, 512, NULL, infoLog);
            std::cout << "-> Vertex Shader error ( " << "sdfOctreeRender.comp" << " ):" << std::endl;
            std::cout << infoLog << std::endl;
            return;
        }

        checkForOpenGLErrors();

        delete[] fileShader;

        mRenderProgramId = glCreateProgram();
        glAttachShader(mRenderProgramId, computeShaderId);
        glLinkProgram(mRenderProgramId);

        glUseProgram(mRenderProgramId);

        mPixelToViewLocation = glGetUniformLocation(mRenderProgramId, "pixelToView");
        mNearPlaneHalfSizeLocation = glGetUniformLocation(mRenderProgramId, "nearPlaneHalfSize");
        mNearAndFarLocation = glGetUniformLocation(mRenderProgramId, "nearAndFarPlane");
        mInvViewModelMatrixLocation = glGetUniformLocation(mRenderProgramId, "invViewModelMatrix");
        mStartGridSizeLocation = glGetUniformLocation(mRenderProgramId, "startGridSize");
        mDistanceScaleLocation = glGetUniformLocation(mRenderProgramId, "distanceScale");
        mOctreeMinBorderValueLocation = glGetUniformLocation(mRenderProgramId, "minBorderValue");


        mEpsilonLocation = glGetUniformLocation(mRenderProgramId, "epsilon");

        //Options
        mUseAOLocation = glGetUniformLocation(mRenderProgramId, "useAO");
        mUseShadowsLocation = glGetUniformLocation(mRenderProgramId, "useShadows");
        mUseSoftShadowsLocation = glGetUniformLocation(mRenderProgramId, "useSoftShadows");
        mOverRelaxationLocation = glGetUniformLocation(mRenderProgramId, "overRelaxation");
        mUseItColorModeLocation = glGetUniformLocation(mRenderProgramId, "useItColorMode");
        mMaxColorIterationsLocation = glGetUniformLocation(mRenderProgramId, "maxColorIterations");
        mMaxIterationsLocation = glGetUniformLocation(mRenderProgramId, "maxIterations");
        mMaxShadowIterationsLocation = glGetUniformLocation(mRenderProgramId, "maxShadowIterations");
        mDrawLightsLocation =  glGetUniformLocation(mRenderProgramId, "drawLights");
        mRaymarchVersionLocation = glGetUniformLocation(mRenderProgramId, "raymarchVersion");
        mV1TriCubicLocation = glGetUniformLocation(mRenderProgramId, "v1TriCubic");
        mUseTricubicNormalsLocation = glGetUniformLocation(mRenderProgramId, "useTricubicNormals");
        //Lighting
        mLightNumberLocation = glGetUniformLocation(mRenderProgramId, "lightNumber");
        mLightPosLocation = glGetUniformLocation(mRenderProgramId, "lightPos");
        mLightColorLocation = glGetUniformLocation(mRenderProgramId, "lightColor");
        mLightIntensityLocation = glGetUniformLocation(mRenderProgramId, "lightIntensity");
        mLightRadiusLocation = glGetUniformLocation(mRenderProgramId, "lightRadius");

        //Material
        mMetallicLocation = glGetUniformLocation(mRenderProgramId, "matMetallic");
        mRoughnessLocation = glGetUniformLocation(mRenderProgramId, "matRoughness");
        mAlbedoLocation = glGetUniformLocation(mRenderProgramId, "matAlbedo");
        mF0Location = glGetUniformLocation(mRenderProgramId, "matF0");

        checkForOpenGLErrors();
    }

    // CreateTexture
    {
        glGenTextures(1, &mRenderTexture);
        glBindTexture(GL_TEXTURE_2D, mRenderTexture);
        // set the texture wrapping/filtering options (on the currently bound texture object)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        mRenderTextureSize = Window::getCurrentWindow().getWindowSize();
        std::vector<uint32_t> colorImage(mRenderTextureSize.x * mRenderTextureSize.y);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mRenderTextureSize.x, mRenderTextureSize.y, 0, GL_RGBA, GL_FLOAT, NULL);

        glBindImageTexture(0, mRenderTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    }

    // Set octree data
    {
        // Set octree trilinear data
        glGenBuffers(1, &mOctreeSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, mOctreeSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, mInputOctree->getOctreeData().size() * sizeof(OctreeSdf::OctreeNode), mInputOctree->getOctreeData().data(), GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mOctreeSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        mOctreeMatrix = glm::scale(glm::mat4(1.0f), 1.0f / mInputOctree->getGridBoundingBox().getSize()) * glm::translate(glm::mat4(1.0f), -mInputOctree->getGridBoundingBox().min);
        mOctreeDistanceScale = 1.0f / mInputOctree->getGridBoundingBox().getSize().x;
        mOctreeMinBorderValue = mInputOctree->getOctreeMinBorderValue();
    }

    // Set octree tricubic data
    {
        // Set octree data
        glGenBuffers(1, &mOctreeTricubicSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, mOctreeTricubicSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, mInputTricubicOctree->getOctreeData().size() * sizeof(OctreeSdf::OctreeNode), mInputTricubicOctree->getOctreeData().data(), GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mOctreeTricubicSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }



    if (mFirstLoad) {
        // Set plane render
        std::shared_ptr<Mesh> planeMesh = PrimitivesFactory::getPlane();
        planeMesh->applyTransform(glm::scale(glm::mat4(1.0f), glm::vec3(2.0f)));
        mRenderMesh.start();
        mRenderMesh.setIndexData(planeMesh->getIndices());
        mRenderMesh.setVertexData(std::vector<RenderMesh::VertexParameterLayout> {
                                            RenderMesh::VertexParameterLayout(GL_FLOAT, 3)
                                    }, planeMesh->getVertices().data(), planeMesh->getVertices().size());
        mFirstLoad = false;
    }

    screenPlaneShader.setInputTexture(mRenderTexture);
    mRenderMesh.setShader(&screenPlaneShader);

    mOctreeStartGridSize = mInputOctree->getStartGridSize();
    float minNodeSize = mInputOctree->getSampleArea().getSize().x / float(1 << mInputOctree->getOctreeMaxDepth());
    mEpsilon = minNodeSize / 2048.0f;
    mEpsilon10000 = mEpsilon * 10000;
    mInputOctree = nullptr; // We do not need the octree in the CPU any more
    mInputTricubicOctree = nullptr;
}

void RenderSdf::draw(Camera* camera)
{
    glm::ivec2 currentScreenSize = Window::getCurrentWindow().getWindowSize();

    if( currentScreenSize.x != mRenderTextureSize.x ||
        currentScreenSize.y != mRenderTextureSize.y)
    {
        mRenderTextureSize = Window::getCurrentWindow().getWindowSize();
        glBindTexture(GL_TEXTURE_2D, mRenderTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mRenderTextureSize.x, mRenderTextureSize.y, 0, GL_RGBA, GL_FLOAT, NULL);
        glBindImageTexture(0, mRenderTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mRenderTexture);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mOctreeSSBO);

    glUseProgram(mRenderProgramId);

    const glm::vec2 nearAndFarPlane = glm::vec2(camera->getZNear(), camera->getZFar());
    const glm::mat4x4& invViewMat = mOctreeMatrix * camera->getInverseViewMatrix();
    float screenHalfSize = 0.5f * glm::tan(glm::radians(camera->getFov())) * nearAndFarPlane.x;
    float screenHalfSizeAspectRatio = screenHalfSize * camera->getRatio();
    glm::vec2 nearPlaneHalfSize = glm::vec2(screenHalfSizeAspectRatio, screenHalfSize);

    // Set camera configuration
    glUniform2f(mPixelToViewLocation, 2.0f * nearPlaneHalfSize.x / static_cast<float>(mRenderTextureSize.x), 
                                      2.0f * nearPlaneHalfSize.y / static_cast<float>(mRenderTextureSize.y));
    glUniform2f(mNearPlaneHalfSizeLocation, nearPlaneHalfSize.x, nearPlaneHalfSize.y);
    glUniform2f(mNearAndFarLocation, nearAndFarPlane.x, nearAndFarPlane.y);
    glUniformMatrix4fv(mInvViewModelMatrixLocation, 1, GL_FALSE, glm::value_ptr(invViewMat));
    glUniform3f(mStartGridSizeLocation, mOctreeStartGridSize.x, mOctreeStartGridSize.y, mOctreeStartGridSize.z);
    glUniform1f(mDistanceScaleLocation, mOctreeDistanceScale);
    glUniform1f(mOctreeMinBorderValueLocation, mOctreeMinBorderValue);

    //mEpsilon = 0.5f*(2.0f/mRenderTextureSize.x); //radius of a pixel in screen space
    //mEpsilon = 0.0001f;
    if (mEpsilon != mEpsilon10000/10000) mEpsilon = mEpsilon10000/10000;
    glUniform1f(mEpsilonLocation, mEpsilon);
    //Options
    glUniform1i(mUseAOLocation, mUseAO);
    glUniform1i(mUseShadowsLocation, mUseShadows);
    glUniform1i(mUseSoftShadowsLocation, mUseSoftShadows);
    glUniform1f(mOverRelaxationLocation, mOverRelaxation);
    glUniform1i(mUseItColorModeLocation, mUseItColorMode);
    glUniform1i(mMaxIterationsLocation, mMaxIterations);
    glUniform1i(mMaxColorIterationsLocation, mMaxColorIterations);
    glUniform1i(mMaxShadowIterationsLocation, mMaxShadowIterations);
    glUniform1i(mDrawLightsLocation, mDrawLights);
    glUniform1i(mRaymarchVersionLocation, mRaymarchVersion);
    glUniform1i(mV1TriCubicLocation, mV1TriCubic);
    glUniform1i(mUseTricubicNormalsLocation, mUseTricubicNormals);
    //Lighting
    glUniform1i(mLightNumberLocation, mLightNumber);
    glUniform3fv(mLightPosLocation, 4, glm::value_ptr(mLightPosition[0]));
    glUniform3fv(mLightColorLocation, 4, glm::value_ptr(mLightColor[0]));
    glUniform1fv(mLightIntensityLocation, 4, &mLightIntensity[0]);
    glUniform1fv(mLightRadiusLocation, 4, &mLightRadius[0]);

    //Material
    glUniform1f(mMetallicLocation, mMetallic);
    glUniform1f(mRoughnessLocation, mRoughness);
    glUniform3f(mAlbedoLocation, mAlbedo.x, mAlbedo.y, mAlbedo.z);
    glUniform3f(mF0Location, mF0.x, mF0.y, mF0.z);

    glDispatchCompute(mRenderTextureSize.x/16, mRenderTextureSize.y/16, 1);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    mRenderMesh.draw(camera);
}

void RenderSdf::drawGui()
{
    if (ImGui::BeginMainMenuBar()) 
    {
        if (ImGui::BeginMenu("Scene")) 
        {
            ImGui::MenuItem("Show scene settings", NULL, &mShowSceneGUI);	
            ImGui::MenuItem("Show lighting settings", NULL, &mShowLightingGUI);
            ImGui::MenuItem("Show algorithm settings", NULL, &mShowAlgorithmGUI);	
            ImGui::MenuItem("Show model settings", NULL, &mShowSdfModelGUI);	
            
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    if (mShowSceneGUI) 
    {
        ImGui::Begin("Scene");
        ImGui::Text("Scene Settings");
        //ImGui::Checkbox("Draw Plane", &mDrawPlane);
        //if (mDrawPlane) ImGui::SliderFloat("Plane Position", &mPlanePos, -1.0f, 1.0f);
        ImGui::Checkbox("AO", &mUseAO);
        ImGui::Checkbox("Shadows", &mUseShadows);
        if (mUseShadows) ImGui::Checkbox("Soft Shadows", &mUseSoftShadows);

        ImGui::End();
    }

    if (mShowLightingGUI)
    {
        ImGui::Begin("Lighting settings");
        ImGui::Checkbox("Draw Lights", &mDrawLights);
        ImGui::SliderInt("Lights", &mLightNumber, 1, 4);

        for (int i = 0; i < mLightNumber; ++i) { //DOES NOT WORK, PROBLEM WITH REFERENCES
            ImGui::Text("Light %d", i);
            std::string pos = "Position##"+std::to_string(i+48);
            std::string col = "Color##"+std::to_string(i+48);
            std::string intens = "Intensity##"+std::to_string(i+48);
            std::string radius = "Radius##"+std::to_string(i+48);
            ImGui::InputFloat3(pos.c_str(), reinterpret_cast<float*>(&mLightPosition[i]));
            ImGui::ColorEdit3(col.c_str(), reinterpret_cast<float*>(&mLightColor[i]));
            ImGui::SliderFloat(intens.c_str(), &mLightIntensity[i], 0.0f, 20.0f);
            ImGui::SliderFloat(radius.c_str(), &mLightRadius[i], 0.01f, 1.0f);
        }

        ImGui::End();
    }

    if (mShowSdfModelGUI)
    {
        ImGui::Begin("Model Settings");
        ImGui::Text("Material");
        ImGui::InputFloat("Metallic", &mMetallic, 0.0f, 1.0f);
        ImGui::InputFloat("Roughness", &mRoughness, 0.0f, 1.0f);
        ImGui::ColorEdit3("Albedo", reinterpret_cast<float*>(&mAlbedo));
        ImGui::ColorEdit3("F0", reinterpret_cast<float*>(&mF0));
        ImGui::End();
    }

    if (mShowAlgorithmGUI) 
    {
        ImGui::Begin("Algorithm Settings");
        ImGui::SliderInt("Version", &mRaymarchVersion, 1, 3);
        if (mRaymarchVersion == 1) ImGui::Checkbox("V1 Tricubic", &mV1TriCubic);
        if (mRaymarchVersion != 1 || mRaymarchVersion == 1 && !mV1TriCubic) ImGui::Checkbox("Use Tricubic Normals", &mUseTricubicNormals);
        if (mRaymarchVersion == 1 && mV1TriCubic && !mUseTricubicNormals) mUseTricubicNormals = true;
        ImGui::InputInt("Max Iterations", &mMaxIterations);
        ImGui::InputInt("Max Shadow Iterations", &mMaxShadowIterations);
        ImGui::Checkbox("Iteration Based Color", &mUseItColorMode);
        if (mUseItColorMode) ImGui::InputInt("Max Color Iterations", &mMaxColorIterations);
        ImGui::SliderFloat("Over Relaxation", &mOverRelaxation, 1.0f, 2.0f);
        ImGui::SliderFloat("Epsilon * 10000", &mEpsilon10000, 0.001f, 21.0f);
        ImGui::End();
    }
}