
def glize(node, light, intensity, color):
    model = node.transformation.astype(numpy.float32)
    for mesh in node.meshes:
        
        material = dict(mesh.material.properties.items())
        texture = material['file'][2:]

        texture_surface = pygame.image.load(texture)
        texture_data = pygame.image.tostring(texture_surface,"RGB",1)
        width = texture_surface.get_width()
        height = texture_surface.get_height()
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        vertex_data = numpy.hstack((
            numpy.array(mesh.vertices, dtype=numpy.float32),
            numpy.array(mesh.normals, dtype=numpy.float32),
            numpy.array(mesh.texturecoords[0], dtype=numpy.float32)
        ))

        faces = numpy.hstack(
            numpy.array(mesh.faces, dtype=numpy.int32)
        )

        vertex_buffer_object = glGenVertexArrays(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, False, 9 * 4, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 9 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, False, 9 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        element_buffer_object = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

        # Cargar matriz de modelo
        glUniformMatrix4fv(
            glGetUniformLocation(shader, "model"), 1 , GL_FALSE, 
            model
        )

        # Cargar matriz de vista
        glUniformMatrix4fv(
            glGetUniformLocation(shader, "view"), 1 , GL_FALSE, 
            glm.value_ptr(view)
        )

        # Cargar matriz de proyeccion
        glUniformMatrix4fv(
            glGetUniformLocation(shader, "projection"), 1 , GL_FALSE, 
            glm.value_ptr(projection)
        )

        # Cargar color difuso de material, alterando la luz y colores RGB.
        diffuse = mesh.material.properties["diffuse"]
        newColor = [i * intensity for i in diffuse]
        
        glUniform4f(
            glGetUniformLocation(shader, "color"),
            newColor[0] + color[0],
            newColor[1] + color[1],
            newColor[2] + color[2],
            1
        )

        # Cargar luz
        glUniform4f(
            glGetUniformLocation(shader, "light"), 
            *light, 1
        )

        glDrawElements(GL_TRIANGLES, len(faces), GL_UNSIGNED_INT, None)

        # glBindBuffer(0, element_buffer_object)


    for child in node.children:
        glize(child, light, intensity, color)

def process_input(radio, angulo, centro, light, intensity, color):

    for event in pygame.event.get():
        # Salir del programa
        if event.type == pygame.QUIT:
            return True

        if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
            return True

        # Opciones para presionar teclas:
        if event.type == pygame.KEYDOWN:

            # Tecla flecha izquierda
            if event.key == pygame.K_LEFT:
                angulo += 5
                camera.x = moverX(radio, angulo, centro)
                camera.z = moverY(radio, angulo, centro)

            # Tecla flecha derecha
            if event.key == pygame.K_RIGHT:
                angulo -= 5
                camera.x = moverX(radio, angulo, centro)
                camera.z = moverY(radio, angulo, centro)

            # Tecla W
            if event.key == pygame.K_w:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

            # Tecla F
            if event.key == pygame.K_f:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Tecla P
            if event.key == pygame.K_p:
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)

            # Tecla flecha arriba
            if event.key == pygame.K_UP:
                if camera.y < 210:
                    camera.y += camera_speed

            # Tecla flecha abajo
            if event.key == pygame.K_DOWN:
                if camera.y > -70:
                    camera.y -= camera_speed

            # Tecla signo +
            if event.key == pygame.K_KP_PLUS:
                if radio >= 100:
                    radio -= 5
                    camera.x = moverX(radio, angulo, centro)
                    camera.z = moverY(radio, angulo, centro)

            # Tecla signo -
            if event.key == pygame.K_KP_MINUS:
                if radio <= 300:
                    radio += 5
                    camera.x = moverX(radio, angulo, centro)
                    camera.z = moverY(radio, angulo, centro)
                    
            # Tecla L
            if event.key == pygame.K_l:
                if intensity < 6:
                    intensity += 0.5    

            # Tecla D
            if event.key == pygame.K_d:
                if intensity > 0:
                    intensity -= 0.5

            # Tecla R
            if event.key == pygame.K_r:
                color.x += 0.05

            # Tecla G
            if event.key == pygame.K_g:
                color.y += 0.05

            # Tecla B
            if event.key == pygame.K_b:
                color.z += 0.05

            # Tecla O
            if event.key == pygame.K_o:
                color.x, color.y, color.z = 0, 0, 0

            # Tecla no. 1
            if event.key == pygame.K_1:
                # Cara 1
                light.x, light.y, light.z = -100, 100, 0
                
            # Tecla no. 2
            if event.key == pygame.K_2:
                # Cara 2
                light.x, light.y, light.z = 0, 100, 100

            # Tecla no. 3
            if event.key == pygame.K_3:
                # Cara 3
                light.x, light.y, light.z = 0, 100, -100
                
            # Tecla no. 4
            if event.key == pygame.K_4:
                # Cara 4
                light.x, light.y, light.z = 100, 100, 0

            # Tecla no. 5
            if event.key == pygame.K_5:
                # Cara 5
                light.x, light.y, light.z = -100, 100, 100

            # Tecla no. 6
            if event.key == pygame.K_6:
                # Cara 6
                light.x, light.y, light.z = -100, 100, -100

            # Tecla no. 7
            if event.key == pygame.K_7:
                # Cara 7
                light.x, light.y, light.z = 100, 100, 100

            # Tecla no. 8
            if event.key == pygame.K_8:
                # Cara 8
                light.x, light.y, light.z = 100, 100, -100

    return False, angulo, radio, light, intensity, color

# Posicion en X, dependiendo de radio, angulo y centro de circulo.
def moverX(radio, angulo, centro):
    return radio * math.cos(math.radians(angulo)) + centro

# Posicion en Y, dependiendo de radio, angulo y centro de circulo.
def moverY(radio, angulo, centro):
    return radio * math.sin(math.radians(angulo)) + centro

# -----------------------------------------------------------------------

import pygame
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import glm
import pyassimp
import numpy
import math

# Inicializacion
pygame.init()
pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.time.Clock()
pygame.key.set_repeat(1, 10)

# Color de fondo y profundidad
glClearColor(0.18, 0.18, 0.18, 1.0)
glEnable(GL_DEPTH_TEST)
glEnable(GL_TEXTURE_2D)

# Codigo de shader: vertex y fragment.
vertex_shader = """
#version 330

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 normal;
layout (location = 2) in vec2 texcoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec4 color;
uniform vec4 light;

out vec4 vertexColor;
out vec2 vertexTexcoords;

void main()
{
    float intensity = dot(normal, normalize(light - position));
    gl_Position = projection * view * model * position;
    vertexColor = color * intensity;
    vertexTexcoords = texcoords;
}
"""

fragment_shader = """
#version 330

layout (location = 0) out vec4 diffuseColor;

in vec4 vertexColor;
in vec2 vertexTexcoords;

uniform sampler2D tex;

void main()
{
    diffuseColor = vertexColor * texture(tex, vertexTexcoords);
}
"""

# Se compilan los shaders.
shader = shaders.compileProgram(
    shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
    shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
)
glUseProgram(shader)


# Matrices modelo, vista, proyeccion y viewpor.
model = glm.mat4(1)
view = glm.mat4(1)
projection = glm.perspective(glm.radians(45), 800/600, 0.1, 1000.0)

glViewport(0, 0, 800, 600)

# Se carga el modelo 3D.
scene = pyassimp.load('10067_Eiffel_Tower_v1_max2010_it1.obj')

# Inicializacion de variables para utilizar en el programa.
light = glm.vec3(-100, 100, 0)
camera = glm.vec3(0, 0, 200)
color = glm.vec3(0, 0, 0)

camera_speed = 10
radio = 200
angulo = 180
centro = 0
intensity = 1.5

inicio = True

print("""
Las teclas que puede utilizar son:
	--> Flecha izquierda: Girar el modelo en sentido horario.-
	--> Flecha derecha: Girar el modelo en sentido antihorario.
	--> Flecha arriba: Subir la vista.
	--> Flecha abajo: Bajar la vista.
	--> Tecla W: Convertir el modelo un renderizado de lineas.
	--> Tecla f: Convertir el modelo un renderizado normal.
	--> Tecla p: Convertir el modelo un renderizado de puntos.
	--> Tecla +: Realizar zoom-in al modelo.
	--> Tecla -: Realizar zoom-out al modelo.
	--> Tecla l: Subir la intensidad de la luz.
	--> Tecla d: Bajar la intensidad de la luz.
	--> Tecla r: Subir el valor R del conjunto RGB.
	--> Tecla g: Subir el valor G del conjunto RGB.
	--> Tecla b: Subir el valor B del conjunto RGB.
	--> Tecla o: Regresar el color a lo original.
	--> Teclas numericas del 1 al 8: Posicionar la luz en diferentes puntos (diferentes caras del modelo).
""")

# Renderizacion de modelo
done = False
while not done:
    # Posicionar el modelo a un punto original.
    if inicio:
        camera.x = moverX(radio, angulo, centro)
        camera.z = moverY(radio, angulo, centro)
        inicio = False
    
    glClearColor(0.6, 0.6, 0.6, 0.6)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    view = glm.lookAt(camera, glm.vec3(0, 20, 0), glm.vec3(0, 1, 0))

    glize(scene.rootnode, light, intensity, color)

    glColor3f(1.0, 0.0, 0.0)

    done, angulo, radio, light, intensity, color = process_input(radio, angulo, centro, light, intensity, color)
    clock.tick(15)
    pygame.display.flip()



