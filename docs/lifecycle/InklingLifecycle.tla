---------------------------- MODULE InklingLifecycle ----------------------------
\* The inkling command-line tool: `weights` (download), `check-weights` (verify), `kernels` (build) and `serve`.
\* Up to two terminals run commands at the same time, the user can edit files at any time, and a command
\* can be interrupted. The rules below are what the tool must guarantee.
\*   UseLock       the commands run one at a time (a lock file held from the first check to the end)
\*   CheckPrereqs  `serve` refuses unless the weights are verified and the kernels are fresh, and checks
\*                 again right before it starts the server
EXTENDS FiniteSets, Naturals

CONSTANTS Invokers, UseLock, CheckPrereqs, None

VARIABLES weights,   \* "absent", "partial", "present" (complete, not verified) or "verified"
          wdirty,    \* the weight files changed since the current or last verification began
          kernels,   \* "absent", "building", "fresh" (built from the current sources) or "stale"
          kdirty,    \* the kernel sources changed since the current or last build began
          lock,      \* the invocation holding the lock, or None
          cmd,       \* per invocation: the command it runs, or "none"
          ph,        \* per invocation: "idle", "wait", "run", "killed", "load" or "work"
          procs,     \* running server processes (named by the invocation that launched them)
          loading,   \* servers still reading the weights and kernels
          okAtLaunch \* servers that were launched with verified weights and fresh kernels

vars == <<weights, wdirty, kernels, kdirty, lock, cmd, ph, procs, loading, okAtLaunch>>

Cmds == {"serve", "build", "download", "verify"}
Prereqs == weights = "verified" /\ kernels = "fresh"
Past == {"run", "killed", "load", "work"}      \* phases after the lock is taken

Init == /\ weights = "absent" /\ wdirty = FALSE
        /\ kernels = "absent" /\ kdirty = FALSE
        /\ lock = None
        /\ cmd = [i \in Invokers |-> "none"]
        /\ ph = [i \in Invokers |-> "idle"]
        /\ procs = {} /\ loading = {} /\ okAtLaunch = {}

\* A command ends: it gives up the lock and the invocation is idle again.
Finish(i) == /\ cmd' = [cmd EXCEPT ![i] = "none"]
             /\ ph' = [ph EXCEPT ![i] = "idle"]
             /\ lock' = IF lock = i THEN None ELSE lock

Start(i, c) == /\ ph[i] = "idle"
               /\ cmd' = [cmd EXCEPT ![i] = c]
               /\ ph' = [ph EXCEPT ![i] = "wait"]
               /\ UNCHANGED <<weights, wdirty, kernels, kdirty, lock, procs, loading, okAtLaunch>>

Enter(i) == /\ ph[i] = "wait"
            /\ IF UseLock THEN lock = None /\ lock' = i ELSE UNCHANGED lock
            /\ ph' = [ph EXCEPT ![i] = "run"]
            /\ UNCHANGED <<weights, wdirty, kernels, kdirty, cmd, procs, loading, okAtLaunch>>

\* ---- serve: check, stop the running server, check again, start the new one, wait until it is ready
ServeRefuseEarly(i) == /\ ph[i] = "run" /\ cmd[i] = "serve" /\ CheckPrereqs /\ ~Prereqs
                       /\ Finish(i)
                       /\ UNCHANGED <<weights, wdirty, kernels, kdirty, procs, loading, okAtLaunch>>

ServeStop(i) == /\ ph[i] = "run" /\ cmd[i] = "serve" /\ (~CheckPrereqs \/ Prereqs)
                /\ procs' = {} /\ loading' = {} /\ okAtLaunch' = {}
                /\ ph' = [ph EXCEPT ![i] = "killed"]
                /\ UNCHANGED <<weights, wdirty, kernels, kdirty, lock, cmd>>

ServeRefuseLate(i) == /\ ph[i] = "killed" /\ cmd[i] = "serve" /\ CheckPrereqs /\ ~Prereqs
                      /\ Finish(i)
                      /\ UNCHANGED <<weights, wdirty, kernels, kdirty, procs, loading, okAtLaunch>>

ServeLaunch(i) == /\ ph[i] = "killed" /\ cmd[i] = "serve" /\ (~CheckPrereqs \/ Prereqs)
                  /\ procs' = procs \cup {i}
                  /\ loading' = loading \cup {i}
                  /\ okAtLaunch' = IF Prereqs THEN okAtLaunch \cup {i} ELSE okAtLaunch
                  /\ ph' = [ph EXCEPT ![i] = "load"]
                  /\ UNCHANGED <<weights, wdirty, kernels, kdirty, lock, cmd>>

ServeReady(i) == /\ ph[i] = "load" /\ cmd[i] = "serve"
                 /\ loading' = loading \ {i}
                 /\ Finish(i)
                 /\ UNCHANGED <<weights, wdirty, kernels, kdirty, procs, okAtLaunch>>

ServeLoadFails(i) == /\ ph[i] = "load" /\ cmd[i] = "serve"
                     /\ procs' = procs \ {i} /\ loading' = loading \ {i} /\ okAtLaunch' = okAtLaunch \ {i}
                     /\ Finish(i)
                     /\ UNCHANGED <<weights, wdirty, kernels, kdirty>>

\* A ready server stops (Ctrl+C, a crash, the user closes it).
ServerExits(s) == /\ s \in procs /\ s \notin loading
                  /\ procs' = procs \ {s} /\ okAtLaunch' = okAtLaunch \ {s}
                  /\ UNCHANGED <<weights, wdirty, kernels, kdirty, lock, cmd, ph, loading>>

\* ---- kernels: the stamp that marks them fresh is written last, and only if the sources did not change
BuildStart(i) == /\ ph[i] = "run" /\ cmd[i] = "build"
                 /\ kernels' = "building" /\ kdirty' = FALSE
                 /\ ph' = [ph EXCEPT ![i] = "work"]
                 /\ UNCHANGED <<weights, wdirty, lock, cmd, procs, loading, okAtLaunch>>

BuildDone(i) == /\ ph[i] = "work" /\ cmd[i] = "build"
                /\ kernels' = IF kdirty THEN "stale" ELSE "fresh"
                /\ Finish(i)
                /\ UNCHANGED <<weights, wdirty, kdirty, procs, loading, okAtLaunch>>

\* ---- weights: download needs no running server; verification records the files it saw
DownloadRefuse(i) == /\ ph[i] = "run" /\ cmd[i] = "download" /\ procs # {}
                     /\ Finish(i)
                     /\ UNCHANGED <<weights, wdirty, kernels, kdirty, procs, loading, okAtLaunch>>

DownloadStart(i) == /\ ph[i] = "run" /\ cmd[i] = "download" /\ procs = {}
                    /\ weights' = "partial" /\ wdirty' = TRUE
                    /\ ph' = [ph EXCEPT ![i] = "work"]
                    /\ UNCHANGED <<kernels, kdirty, lock, cmd, procs, loading, okAtLaunch>>

DownloadDone(i) == /\ ph[i] = "work" /\ cmd[i] = "download"
                   /\ weights' = "present"
                   /\ Finish(i)
                   /\ UNCHANGED <<wdirty, kernels, kdirty, procs, loading, okAtLaunch>>

VerifyRefuse(i) == /\ ph[i] = "run" /\ cmd[i] = "verify" /\ weights \in {"absent", "partial"}
                   /\ Finish(i)
                   /\ UNCHANGED <<weights, wdirty, kernels, kdirty, procs, loading, okAtLaunch>>

VerifyStart(i) == /\ ph[i] = "run" /\ cmd[i] = "verify" /\ weights \in {"present", "verified"}
                  /\ wdirty' = FALSE
                  /\ ph' = [ph EXCEPT ![i] = "work"]
                  /\ UNCHANGED <<weights, kernels, kdirty, lock, cmd, procs, loading, okAtLaunch>>

VerifyMatches(i) == /\ ph[i] = "work" /\ cmd[i] = "verify"
                    /\ weights' = IF wdirty THEN "present" ELSE "verified"
                    /\ Finish(i)
                    /\ UNCHANGED <<wdirty, kernels, kdirty, procs, loading, okAtLaunch>>

VerifyDiffers(i) == /\ ph[i] = "work" /\ cmd[i] = "verify"
                    /\ weights' = IF weights = "verified" THEN "present" ELSE weights
                    /\ Finish(i)
                    /\ UNCHANGED <<wdirty, kernels, kdirty, procs, loading, okAtLaunch>>

\* A command is interrupted (Ctrl+C, a crash, the machine restarts): the operating system releases the lock.
Interrupt(i) == /\ ph[i] # "idle"
                /\ kernels' = IF cmd[i] = "build" /\ ph[i] = "work" THEN "stale" ELSE kernels
                /\ procs' = IF ph[i] = "load" THEN procs \ {i} ELSE procs
                /\ loading' = IF ph[i] = "load" THEN loading \ {i} ELSE loading
                /\ okAtLaunch' = IF ph[i] = "load" THEN okAtLaunch \ {i} ELSE okAtLaunch
                /\ Finish(i)
                /\ UNCHANGED <<weights, wdirty, kdirty>>

\* The user changes files whenever they like.
EditSources == /\ kdirty' = TRUE
               /\ kernels' = IF kernels = "fresh" THEN "stale" ELSE kernels
               /\ UNCHANGED <<weights, wdirty, lock, cmd, ph, procs, loading, okAtLaunch>>

ChangeWeights == /\ wdirty' = TRUE
                 /\ weights' = IF weights = "verified" THEN "present" ELSE weights
                 /\ UNCHANGED <<kernels, kdirty, lock, cmd, ph, procs, loading, okAtLaunch>>

Next == \/ \E i \in Invokers :
              \/ \E c \in Cmds : Start(i, c)
              \/ Enter(i)
              \/ ServeRefuseEarly(i) \/ ServeStop(i) \/ ServeRefuseLate(i) \/ ServeLaunch(i)
              \/ ServeReady(i) \/ ServeLoadFails(i)
              \/ BuildStart(i) \/ BuildDone(i)
              \/ DownloadRefuse(i) \/ DownloadStart(i) \/ DownloadDone(i)
              \/ VerifyRefuse(i) \/ VerifyStart(i) \/ VerifyMatches(i) \/ VerifyDiffers(i)
              \/ Interrupt(i)
        \/ \E s \in Invokers : ServerExits(s)
        \/ EditSources \/ ChangeWeights

Spec == Init /\ [][Next]_vars

\* ---- Rules
TypeOK == /\ weights \in {"absent", "partial", "present", "verified"}
          /\ kernels \in {"absent", "building", "fresh", "stale"}
          /\ wdirty \in BOOLEAN /\ kdirty \in BOOLEAN
          /\ lock \in Invokers \cup {None}
          /\ cmd \in [Invokers -> Cmds \cup {"none"}]
          /\ ph \in [Invokers -> {"idle", "wait"} \cup Past]
          /\ procs \subseteq Invokers /\ loading \subseteq procs /\ okAtLaunch \subseteq procs

\* There is never more than one server, so `serve` replaces the running one.
SingleServer == Cardinality(procs) <= 1

\* A server only ever starts with verified weights and fresh kernels.
ServersStartedWithPrereqs == procs \subseteq okAtLaunch

\* While a server reads the weights and kernels, nothing rewrites them.
NoWriteWhileLoading == loading # {} => (weights # "partial" /\ kernels # "building")

\* "fresh" and "verified" are never claimed for files that changed after the build or the check began.
FreshMeansCurrent == kernels = "fresh" => ~kdirty
VerifiedMeansCurrent == weights = "verified" => ~wdirty

\* Only the holder of the lock is past the check phase.
OnlyLockHolderRuns == UseLock => \A i \in Invokers : ph[i] \in Past => lock = i
=============================================================================
