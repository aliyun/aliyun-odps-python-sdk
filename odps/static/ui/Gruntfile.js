module.exports = function (grunt) {
    grunt.initConfig({
        pkg: grunt.file.readJSON('package.json'),

        copy: {
            main: {
                files: [
                    {expand: true, flatten: true, src: ['src/echarts.min.js'], dest: 'target/build/', filter: 'isFile'},
                    {expand: true, flatten: true, src: ['src/*.png'], dest: 'target/', filter: 'isFile'},
                    {expand: true, flatten: true, src: ['src/fonts/*'], dest: 'target/fonts', filter: 'isFile'}
                ]
            }
        },

        uglify: {
            options: {
                banner: '/*! <%= pkg.name %> <%= grunt.template.today("yyyy-mm-dd") %> */\n'
            },
            my_target: {
                files: [
                    {
                        expand: true,
                        cwd: 'src/',
                        src: ['**/*.js', '!**/*.min.js'],
                        dest: 'target/build/',
                        rename: function (dest, src) {
                            var folder = src.substring(0, src.lastIndexOf('/'));
                            var filename = src.substring(src.lastIndexOf('/'), src.length);
                            filename = filename.substring(0, filename.lastIndexOf('.'));
                            return dest + folder + filename + '.min.js';
                        }
                    }
                ]
            }
        },

        requirejs: {
            compile: {
                options: {
                    baseUrl: "target/build/",
                    paths: {
                        jquery: "empty:",
                        echarts: "empty:",
                        "jupyter-js-widgets": "empty:",
                    },
                    include: ['chosen.min', 'westeros.min', 'echarts.min', 'common.min', 'html-notify.min',
                        'ml-retry.min', 'progress.min', 'df-view.min'],
                    out: 'target/main.js',
                    optimize: 'none',
                    shim: {
                        'chosen': {
                            deps: [ 'jquery' ],
                            exports: 'jQuery.fn.chosen'
                        }
                    }
                }
            }
        },

        cssmin: {
            options: {
                banner: '/*! <%= pkg.name %> <%= grunt.template.today("yyyy-mm-dd") %> */\n',
                beautify: {
                    ascii_only: true
                }
            },
            my_target: {
                files: [
                    {
                        expand: true,
                        cwd: 'src/',
                        src: '*.css',
                        dest: 'target/'
                    },
                    {
                        expand: true,
                        cwd: 'src/fonts',
                        src: '*.css',
                        dest: 'target/fonts'
                    }
                ]
            }
        },

        clean: ['target/build'],
    });

    grunt.loadNpmTasks('grunt-contrib-copy');
    grunt.loadNpmTasks('grunt-contrib-uglify');
    grunt.loadNpmTasks('grunt-contrib-requirejs');
    grunt.loadNpmTasks('grunt-contrib-cssmin');
    grunt.loadNpmTasks('grunt-contrib-clean');
    grunt.loadNpmTasks('grunt-file-append');

    grunt.registerTask('default', ['copy', 'uglify', 'requirejs', 'cssmin', 'clean']);

};