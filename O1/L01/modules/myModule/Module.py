def sayHello():
    print("Hello from my first module...")

def moreDummyFunc(myParam=True):
    if myParam == True: 
        print("Param true :)")
        return
    
    print("Param untrue :(")

def reloadMe():
    print("I have been reloaded!")
    