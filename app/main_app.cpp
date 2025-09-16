
#include "cutsim_window.hpp"
#include <QApplication>
CutsimWindow *window;
int main(int argc, char *argv[])
{
    QApplication app( argc, argv );
    QStringList qsl;
    for(int i = 1; i < argc; i++) qsl.append(argv[i]);

//    CutsimWindow *window = new CutsimWindow(qsl);
    window = new CutsimWindow(qsl);
    window->show();
    int retval = app.exec();
    delete window;
    return retval;
}

